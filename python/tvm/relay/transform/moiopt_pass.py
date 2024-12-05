import tvm
from tvm import relay
import numpy as np

from ..expr_functor import ExprMutator, ExprVisitor
from .moiopt.graph_analyzer import GraphAnalyzer
from .moiopt.network import Network
from .moiopt.relay_util import ReplaceCallPass, FindCall, findFromOtherModule, normalizePadding
from .moiopt.relay_util import isDepthwiseConv, exprToStr, getCheckedType
from .moiopt.memplanner import MemoryPlanner, memLayoutWithTimeout
from .pathdiscovery import SplitType, PathDiscovery
from .transform import function_pass


def getDominatingBufs(n: Network):
    sched = n.createBestSchedule()
    planner = MemoryPlanner(sched)

    def getCritBufs(memLayout):
        return {buf: None for buf in memLayout.getBufsByCriticality()}

    critBufs = memLayoutWithTimeout(n, planner, getCritBufs)
    return [buf for buf in critBufs]


# Applies the DefuseOps while remembering a given expr.
def defuseAndRememberExpr(mod, expr):
    # Add marker annotation to get expr after defuse.
    mod = ReplaceCallPass(expr, lambda call: relay.annotation.stop_fusion(call))(mod)
    mod = relay.transform.DefuseOps()(mod)

    # Find the marker annotation.
    def isStopFuse(e):
        if isinstance(e, relay.Call):
            if isinstance(e.op, tvm.ir.Op):
                if e.op.name == "annotation.stop_fusion":
                    return True
        return False

    markerExprs = FindCall().run(mod, isStopFuse)
    assert len(markerExprs) == 1

    # Remove marker annotation.
    mod = ReplaceCallPass(markerExprs[0], lambda call: call.args[0])(mod)

    # The ExprMutator did replace expr at this point. Recover it efficiently.
    expr = findFromOtherModule(mod, markerExprs[0].args[0])

    return mod, expr


class ReplaceInfo:
    def __init__(self, call, tupVal, tupIndex, argIndex):
        self.call = call
        self.tupVal = tupVal
        self.tupIndex = tupIndex
        self.argIndex = argIndex
        self.newVar = None


# Lifts TupleGetItem nodes out of funcs into the global scope.
# This eliminates unnecessary dataflow dependencies.
@function_pass(opt_level=0)
class FixTupleDepdendencyPass(ExprMutator):
    def __init__(self):
        super().__init__()
        self.parentCall = None
        self.replaceInfo = []

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            # Remember in which sub-function we are.
            prevParentCall = self.parentCall
            self.parentCall = call
            newCall = super().visit_call(call)
            self.parentCall = prevParentCall

            for info in self.replaceInfo:
                if info.call == call:
                    # Replace this call.

                    # Replace the parameter of the corresponding function.
                    fn = newCall.op
                    params = [p for p in fn.params]
                    params[info.argIndex] = info.newVar
                    newFn = relay.Function(params, fn.body, fn.ret_type, fn.type_params, fn.attrs)

                    # Replace the argument to the corresponding call with the TupleGetItem expr.
                    args = [arg for arg in newCall.args]
                    args[info.argIndex] = relay.TupleGetItem(args[info.argIndex], info.tupIndex)
                    return relay.Call(newFn, args, newCall.attrs, newCall.type_args, newCall.span)

            return newCall
        return super().visit_call(call)

    def visit_tuple_getitem(self, op):
        if self.parentCall != None:
            # Find which function parameter corresponds to the tuple.
            argIndex = None
            for i, param in enumerate(self.parentCall.op.params):
                if param == op.tuple_value:
                    argIndex = i
            if argIndex != None:
                # Remeber the info for func and call replacements.
                info = ReplaceInfo(self.parentCall, op.tuple_value, op.index, argIndex)
                self.replaceInfo.append(info)
                # Replace the TupleGetItem with the accessed sub-field.
                info.newVar = relay.Var(info.tupVal.name_hint, info.tupVal.type_annotation.fields[info.tupIndex])
                return info.newVar
        return super().visit_tuple_getitem(op)


# Returns the offset and size of a partition, accoring to np.array_split.
def getPartitionOffsetAndSize(totalSize, numPartitions, partNum):
    if totalSize < numPartitions:
        raise RuntimeError("too many partitions: " + str(numPartitions) + " (for size: " + str(totalSize) + ")")
    sectionSize, extraSizeSections = divmod(totalSize, numPartitions)
    if partNum < extraSizeSections:
        return (partNum * (sectionSize + 1), sectionSize + 1)
    else:
        offset = extraSizeSections * (sectionSize + 1) + (partNum - extraSizeSections) * sectionSize
        return (offset, sectionSize)


class SplitParams:
    def __init__(self, axis, size, offset):
        self.axis = axis
        self.size = size
        self.offset = offset

    def __repr__(self):
        return "SplitParams(" + str(self.axis) + "," + str(self.size) + "," + str(self.offset) + ")"


def splitConst(const, splitParams, partNum):
    if not isinstance(const, relay.Constant):
        raise RuntimeError("splitConst did not get a Constant")

    if len(splitParams.axes) == 0:
        return const
    elif len(splitParams.axes) == 1:
        splitAx = splitParams.axes[0]
        r = splitAx.ranges[partNum]
        indices = np.arange(r[0], r[1])
        arr = np.take(const.data.asnumpy(), indices, axis=splitAx.axis)
    elif len(splitParams.axes) == 2:
        ax1 = splitParams.axes[0]
        ax2 = splitParams.axes[1]
        r1 = ax1.ranges[partNum // len(ax2.ranges)]
        r2 = ax2.ranges[partNum % len(ax2.ranges)]
        arr = np.take(const.data.asnumpy(), np.arange(r1[0], r1[1]), axis=ax1.axis)
        arr = np.take(arr, np.arange(r2[0], r2[1]), axis=ax2.axis)
    else:
        raise RuntimeError("can not handle 3d split axes")
    return relay.const(tvm.nd.array(arr))


@tvm.ir.transform.module_pass(opt_level=0)
class SplitPathPass(ExprMutator):
    def __init__(self, splitPath):
        super().__init__()

        self.splitPath = splitPath
        self.numPartitions = splitPath.getNumPartitions()

        self.callToReplace = splitPath[-1].expr
        self.callToStopAt = splitPath[0].expr
        self.isUsingImplicitSplit = splitPath[0].isImplicitSplit()
        self.needsMerge = splitPath[-1].isPartial()

        self.splitExprs = None
        self.hintStartAxis = None

    def transform_module(self, mod, ctx):
        func = self.visit(mod["main"])
        # print(func)
        return tvm.IRModule.from_expr(func)

    # After going through one chain of operators in a partition, reset state to the start.
    def resetOperatorWalk(self):
        self.opIndex = 0
        self.currentSplitParams = None

    def advanceOperatorWalk(self):
        self.opIndex += 1

    def getCurrentSplitType(self):
        return self.splitPath[self.opIndex].splitType

    def getCurrentSplitCfg(self):
        return self.splitPath[self.opIndex]

    def getCurrentSplitAxis(self):
        return self.currentSplitParams.axis

    def getCurrentPartitionSize(self):
        return self.currentSplitParams.size

    def setCurrentSplitParams(self, splitParams):
        self.currentSplitParams = splitParams

    def visit_call(self, call):
        if call != self.callToReplace:
            return super().visit_call(call)

        # Insert either CONCAT or MERGE.
        if self.needsMerge:
            return self.insertMerge(call)
        else:
            return self.insertConcat(call)

    def insertMerge(self, call):
        outExprs = self.splitCall(call)

        # Merge partial values from different paths.
        outExpr = outExprs[0]
        for i in range(1, len(outExprs)):
            outExpr = relay.add(outExpr, outExprs[i])

        return outExpr

    def insertConcat(self, call):
        outExprs = self.splitCall(call)
        self.opIndex -= 1
        outSplit = self.getCurrentSplitCfg().outSplit
        if len(outSplit.axes) == 1:
            return relay.concatenate(outExprs, outSplit.axes[0].axis)
        elif len(outSplit.axes) == 2:
            catExprs = []
            width = outSplit.axes[0].getNumPartitions()
            for i in range(0, outSplit.axes[1].getNumPartitions()):
                catExprs.append(relay.concatenate(outExprs[i * width : (i + 1) * width], outSplit.axes[1].axis))
            return relay.concatenate(catExprs, outSplit.axes[0].axis)
        else:
            raise RuntimeError("invalid number of axes")

    def splitCall(self, call):
        outExprs = []
        for i in range(0, self.numPartitions):
            self.resetOperatorWalk()
            splitExpr = self.replaceCall(call, i)
            # Prevent fusion of any consecutive op with the last one in the split path because
            # that would lead to keeping the intermediate buffer alive on multiple split paths.
            outExprs.append(relay.annotation.stop_fusion(splitExpr))
        return outExprs

    def getSplitInput(self, partNum):
        if self.isUsingImplicitSplit:
            return None

        if self.splitExprs == None:
            # Explicit split.
            self.splitExprs = []
            exprToSplit = self.callToStopAt.args[0]
            splitCfg = self.getCurrentSplitCfg()
            for i in range(0, self.numPartitions):
                splitAxes = []
                offsets = []
                endOffsets = []
                strides = []
                for splitAx in splitCfg.inSplit.axes:
                    splitAxes.append(splitAx.axis)
                    r = splitCfg.inSplit.partNumToRange(i, splitAx.axis)
                    offsets.append(r[0])
                    endOffsets.append(r[1])
                    strides.append(1)
                self.splitExprs.append(
                    relay.strided_slice(exprToSplit, offsets, endOffsets, strides=strides, axes=splitAxes)
                )

        return self.splitExprs[partNum]

    def replaceCall(self, call, partNum):
        # Recurse up until the call to stop at so that we process them in consecutive order.
        args = []
        for arg in call.args:
            if isinstance(arg, relay.Call) and call != self.callToStopAt:
                args.append(self.replaceCall(arg, partNum))
            else:
                args.append(arg)

        # Split the input argument.
        if call == self.callToStopAt:
            splitExpr = self.getSplitInput(partNum)
            if splitExpr != None:
                args[0] = splitExpr

        attrs = {}
        if call.attrs != None:
            for attr in call.attrs.keys():
                value = call.attrs[attr]
                attrs[attr] = value

        splitCfg = self.getCurrentSplitCfg()
        for i in range(0, len(args)):
            if not isinstance(args[i], relay.Constant):
                continue
            wSplit = splitCfg.splitWeights(i)
            args[i] = splitConst(args[i], wSplit, partNum)

        # Fix attributes.
        partShape = splitCfg.outSplit.getSplitShape(partNum)
        n = call.op.name

        # Need to cut padding at split boundaries.
        if n == "nn.conv2d" or n == "nn.max_pool2d" or n == "avg_pool2d":
            if splitCfg.splitType == SplitType.FTP:
                padding = normalizePadding(attrs["padding"])
                dataWidthIndex = attrs["data_layout" if n == "nn.conv2d" else "layout"].index("W")
                for splitAx in splitCfg.outSplit.axes:
                    isWidthAxis = splitAx.axis == dataWidthIndex
                    r = splitCfg.outSplit.partNumToRange(partNum, splitAx.axis)
                    if r[0] != 0:
                        padding[1 if isWidthAxis else 0] = 0
                    if r[1] != splitCfg.outSplit.shape[splitAx.axis]:
                        padding[3 if isWidthAxis else 2] = 0
                attrs["padding"] = padding
        elif n == "nn.pad":
            padding = list(attrs["pad_width"])
            for splitAx in splitCfg.outSplit.axes:
                r = splitCfg.outSplit.partNumToRange(partNum, splitAx.axis)
                if r[0] != 0:
                    padding[splitAx.axis] = [0, padding[splitAx.axis][1]]
                if r[1] != splitCfg.outSplit.shape[splitAx.axis]:
                    padding[splitAx.axis] = [padding[splitAx.axis][0], 0]
            attrs["pad_width"] = padding

        if n == "reshape":
            attrs["newshape"] = partShape
        elif n == "nn.conv2d":
            if splitCfg.splitType == SplitType.PARTITIONED:
                assert isDepthwiseConv(call)
                attrs["groups"] = partShape[3]
                attrs["channels"] = partShape[3]
            elif splitCfg.splitType == SplitType.LOP:
                attrs["channels"] = partShape[3]
            elif splitCfg.splitType == SplitType.LIP:
                pass
            elif splitCfg.splitType == SplitType.FTP:
                pass
            else:
                raise RuntimeError(f"unexpected split type for conv: {splitCfg.splitType}")
        elif n in ["nn.contrib_dense_pack", "nn.dense"]:
            if splitCfg.splitType == SplitType.LOP:
                if attrs["units"] is not None:
                    attrs["units"] = partShape[1]
        elif n == "nn.pad":
            # Work around inconsistency of argument definitions of nn.pad
            if len(args) > 1:
                attrs["pad_value"] = args[1]
                args = args[:1]

        self.advanceOperatorWalk()

        return tvm.relay.frontend.common.get_relay_op(n)(*args, **attrs)


# Provides a context for MOIOPT.
class MOIOPTContext:
    def __init__(self, noFTP, onlyFTP, noRecurse, maxPartitions):
        self.noFTP = noFTP
        self.onlyFTP = onlyFTP
        self.noRecurse = noRecurse
        self.firstIter = True
        self.maxPartitions = maxPartitions

    def runAnalysis(self, mod):
        """ return GraphAnalyzer and network.Networkx """
        self.analyzer = GraphAnalyzer()
        self.analyzer.run(mod["main"])
        self.n = self.analyzer.makeNet()
        return self.analyzer, self.n

    def transform(self, mod):
        topAnalyzer, topN = self.runAnalysis(mod)

        largestSize = None
        #Get buffers that are responsible for high memory usage.
        for buf in getDominatingBufs(topN):
            if largestSize == None:
                largestSize = buf.size
            else:
                if buf.size < largestSize * 0.05:
                    continue

            if buf in (topN.getInBufs() + topN.getOutBufs()):
                print("Skipping buf because it is an input or output to the model")
                continue

            print("targetBuf:", buf)

            expr = topAnalyzer.getExprFromBuf(buf)
            resultMod = self.applyFusedTiling(mod, expr)
            if resultMod != None:
                # After the transformation, we need a new analysis before trying again.
                if self.noRecurse:
                    return resultMod
                else:
                    return self.transform(resultMod)

        return mod

    def applyFusedTiling(self, mod, targetExpr):
        mod, targetExpr = defuseAndRememberExpr(mod, targetExpr)
        self.runAnalysis(mod)

        print(mod)

        print("targetExpr:", exprToStr(targetExpr))

        # Find available fusable path.
        pathDiscovery = PathDiscovery(
            targetExpr, self.analyzer, self.n, self.noFTP, self.onlyFTP, self.maxPartitions
        )

        # Select path configuration.
        splitPath = pathDiscovery.discoverBest(mod)
        if splitPath == None:
            return None

        print("Split configuration:", splitPath)

        with open("splitcfglog.txt", "a") as f:
            if not self.firstIter:
                f.write(" + ")
            self.firstIter = False
            f.write(splitPath.veryShortDesc())

        # Transform the relay graph.
        mod = SplitPathPass(splitPath)(mod)
        mod = relay.transform.InferType()(mod)

        # Fuse again for the backend.
        mod = relay.transform.FuseOps()(mod)

        # Fix unwanted tuple dependencies.
        mod = FixTupleDepdendencyPass()(mod)

        print(mod)

        return mod


# Applies the Memory Optimizing Inter-Operator Tiling (MOIOPT).
@tvm.ir.transform.module_pass(opt_level=0)
class MOIOPTPass(ExprMutator):
    def __init__(self, noFTP=False, onlyFTP=False, noRecurse=False, maxPartitions=0):
        self.noFTP = noFTP
        self.onlyFTP = onlyFTP
        self.noRecurse = noRecurse
        self.maxPartitions = int(maxPartitions)

    def transform_module(self, mod, ctx):
        return MOIOPTContext(self.noFTP, self.onlyFTP, self.noRecurse, self.maxPartitions).transform(mod)


def save_toplevel_tuple_info(func: relay.ty.FuncType):
    node_list = []
    tup = func.body
    for f in tup.fields:
        print(f)
        node_list.append(f)
        #assert isinstance(tensor_type, relay.ty.TensorType)
    return node_list


# # TODO at a later point: get also original result scheduled accordingly and function signature
# # Heuristic as the original node has not out_degree=0 Therfore not scheduled to be kept alive till the end
def order_tuple_according_to_num(outputs, best_sched, analyzer):
    """Iterate through schedule and append node if in outputs in ordered list"""
    ordered_tuple_ele = []
    print("Schedule node")
    for op in best_sched.sched:
         print(op)
         if analyzer.OpToExpr[op] in outputs:
            ordered_tuple_ele.append(analyzer.OpToExpr[op])
    return ordered_tuple_ele


# used only one to create new function
def get_new_func_signature(tuple_ele):
    ret_type = []
    for ele in tuple_ele:
        tensor_type = getCheckedType(ele)
        #tensor_type.size
        #tensor_type.dtype
        ret_type.append(tensor_type)
    return ret_type



@tvm.ir.transform.module_pass(opt_level=0)
class MinimizeRAMWithTupleOrder(ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_module(self, mod, ctx):
        top_level_func = mod["main"]
        analyzer = GraphAnalyzer()
        analyzer.run(top_level_func)
        n = analyzer.makeNet()
        best_sched = n.createBestSchedule()

        node_list = save_toplevel_tuple_info(top_level_func)
        #print(f"this is the unordered tuple_list {node_list}")
        ordered_node_list = order_tuple_according_to_num(node_list, best_sched, analyzer)
        #print(f"this is the ordered tuple_list {ordered_node_list}")

        return  tvm.IRModule.from_expr(relay.Function(top_level_func.params, relay.Tuple(ordered_node_list),
        relay.ty.TupleType(get_new_func_signature(ordered_node_list))))
