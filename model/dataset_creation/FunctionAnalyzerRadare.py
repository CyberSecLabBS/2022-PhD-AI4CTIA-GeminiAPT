# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import json
from json import JSONDecodeError
import r2pipe
import networkx as nx
from dataset_creation.BlockFeaturesExtractor import BlockFeaturesExtractor
import warnings
import re
warnings.filterwarnings("ignore")


class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


class RadareFunctionAnalyzer:

    def __init__(self, filename, use_symbol):
        self.r2 = r2pipe.open(filename, flags=['-2'])
        self.filename = filename
        self.arch, _ = self.get_arch()
        self.use_symbol = use_symbol

    def __enter__(self):
        return self

    @staticmethod
    def filter_reg(op):
        return op["value"]

    @staticmethod
    def filter_imm(op):
        imm = int(op["value"])
        if -int(5000) <= imm <= int(5000):
            ret = str(hex(op["value"]))
        else:
            ret = str('HIMM')
        return ret

    @staticmethod
    def filter_mem(op):
        if "base" not in op:
            op["base"] = 0

        if op["base"] == 0:
            r = "[" + "MEM" + "]"
        else:
            reg_base = str(op["base"])
            disp = str(op["disp"])
            scale = str(op["scale"])
            r = '[' + reg_base + "*" + scale + "+" + disp + ']'
        return r

    @staticmethod
    def filter_memory_references(i):
        inst = "" + i["mnemonic"]

        for op in i["opex"]["operands"]:
            if op["type"] == 'reg':
                inst += " " + RadareFunctionAnalyzer.filter_reg(op)
            elif op["type"] == 'imm':
                inst += " " + RadareFunctionAnalyzer.filter_imm(op)
            elif op["type"] == 'mem':
                inst += " " + RadareFunctionAnalyzer.filter_mem(op)
            if len(i["opex"]["operands"]) > 1:
                inst = inst + ","

        if "," in inst:
            inst = inst[:-1]
        inst = inst.replace(" ", "_")

        return str(inst)

    @staticmethod
    def get_callref(my_function, depth):
        calls = {}
        if 'callrefs' in my_function and depth > 0:
            for cc in my_function['callrefs']:
                if cc["type"] == "C":
                    calls[cc['at']] = cc['addr']
        return calls

    def _fix_r2pipe(self):
        output = b""
        # p1, p2 = None, b""
        while True:
            try:
                # starting from pending or pipe itself
                if len(self.r2.pending) > 0:
                    pending = self.r2.pending
                    # p1 = pending
                    self.r2.pending = b""
                else:
                    read = self.r2.process.stdout.read(4096)
                    # p2 += read if read is not None else p2
                    pending = read

                if pending is not None:
                    output += pending
                    zero = output.count(b"\x00")
                    if zero == 1 and output[-1] == 0:
                        # the result is now complete
                        break
                    elif zero == 1 and output[-1] != 0:
                        print(f"[Radare2] There is a null byte in the middle")
                        # print(pending)
                        output = b'[]\n'
                        break
                    elif zero > 1:
                        print(f"[Radare2] There is more than one null byte")
                        # print(pending)
                        output = b'[]\n'
                        break

            except Exception as ex:
                print(f"[Radare2] Exception in _fix_r2pipe {ex}")
                pass

        decoded = output.decode("UTF-8")
        # return p1, p2, output, decoded.strip('\x00')
        return decoded.strip('\x00')

    def process_instructions(self, instructions):
        filtered_instructions = []
        for insn in instructions:
            # operands = []
            if 'opex' not in insn:
                continue
            # for op in insn['opex']['operands']:
            #    operands.append(Dict2Obj(op))
            # insn['operands'] = operands
            stringized = RadareFunctionAnalyzer.filter_memory_references(insn)
            if "x86" in self.arch:
                stringized = "X_" + stringized
            elif "arm" in self.arch:
                stringized = "A_" + stringized
            else:
                stringized = "UNK_" + stringized
            filtered_instructions.append(stringized)
        return filtered_instructions

    def process_block(self, block):
        bytes = ""
        disasm = []
        for op in block['ops']:
            if 'disasm' in op:
                disasm.append(op['disasm'])
                bytes += str(op['bytes'])

        self.r2.cmd("s " + str(block['offset']))
        try:
            instructions = json.loads(self.r2.cmd("aoj " + str(len(block['ops']))))
        except JSONDecodeError:
            payload = self._fix_r2pipe()
            instructions = json.loads(payload)
        except Exception as e:
            print("EXCEPTION 1: {}".format(e))
        try:
            var_izzj = json.loads(self.r2.cmd("izzj"))
            string_addresses = [s['vaddr'] for s in var_izzj]
            bfe = BlockFeaturesExtractor(self.arch, instructions, block['ops'], string_addresses)
            annotations = bfe.getFeatures()
            filtered_instructions = self.process_instructions(instructions)
        except JSONDecodeError:
            # this could happen due to a null byte at the beginning of the buffer
            payload = self._fix_r2pipe()
            string_addresses = json.loads(payload)
            print(string_addresses)
            bfe = BlockFeaturesExtractor(self.arch, instructions, block['ops'], string_addresses)
            annotations = bfe.getFeatures()
            filtered_instructions = self.process_instructions(instructions)
        except Exception as e:
            print("EXCEPTION 2: {}".format(e))
            annotations, filtered_instructions = [], []

        return disasm, bytes, annotations, filtered_instructions

    def function_to_cfg(self, func):
        if self.use_symbol:
            s = 'vaddr'
        else:
            s = 'offset'

        self.r2.cmd('s ' + str(func[s]))
        try:
            var_agfj = self.r2.cmd('agfj ' + str(func[s]))
            cfg = json.loads(var_agfj)
        except JSONDecodeError:
            # this could happen due to a null byte at the beginning of the buffer
            payload = self._fix_r2pipe()
            cfg = json.loads(payload)
        except Exception as e:
            print("EXCEPTION 3: {}".format(e))
            cfg = []

        my_cfg = nx.DiGraph()
        acfg = nx.DiGraph()
        lstm_cfg = nx.DiGraph()

        if len(cfg) == 0:
            return None, None, None
        else:
            cfg = cfg[0]

        for block in cfg['blocks']:
            disasm, block_bytes, annotations, filtered_instructions = self.process_block(block)
            my_cfg.add_node(block['offset'], asm=block_bytes, label=disasm)
            acfg.add_node(block['offset'], features=annotations)
            lstm_cfg.add_node(block['offset'], features=filtered_instructions)

        for block in cfg['blocks']:
            if 'jump' in block:
                if block['jump'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'], block['jump'])
                    acfg.add_edge(block['offset'], block['jump'])
                    lstm_cfg.add_edge(block['offset'], block['jump'])
            if 'fail' in block:
                if block['fail'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'], block['fail'])
                    acfg.add_edge(block['offset'], block['fail'])
                    lstm_cfg.add_edge(block['offset'], block['fail'])

        between = nx.betweenness_centrality(acfg)
        for n in acfg.nodes(data=True):
            d = n[1]['features']
            d['offspring'] = len(nx.descendants(acfg, n[0]))
            d['betweenness'] = between[n[0]]
            n[1]['features'] = d

        return my_cfg, acfg, lstm_cfg

    def get_arch(self):
        try:
            try:
                info = json.loads(self.r2.cmd('ij'))
            except JSONDecodeError:
                # this could happen due to a null byte at the beginning of the buffer
                payload = self._fix_r2pipe()
                info = json.loads(payload)
            except Exception as e:
                print("EXCEPTION 4: {}".format(e))
                info = []
            if 'bin' in info:
                arch = info['bin']['arch']
                bits = info['bin']['bits']

        except Exception as e:
            print(e)
            arch = None
            bits = None
        return arch, bits

    def find_functions(self):
        self.r2.cmd('aa')
        try:
            function_list = json.loads(self.r2.cmd('aflj'))
        except JSONDecodeError:
            # this could happen due to a null byte at the beginning of the buffer
            payload = self._fix_r2pipe()
            function_list = json.loads(payload)
        except Exception as e:
            print("EXCEPTION 5: {}".format(e))
            function_list = []
        return function_list

    def find_functions_by_symbols(self):
        self.r2.cmd('aa')
        try:
            var_isj = self.r2.cmd('isj')
            symbols = json.loads(var_isj)
            fcn_symb = [s for s in symbols if s['type'] == 'FUNC']
        except JSONDecodeError:
            # this could happen due to a null byte at the beginning of the buffer
            payload = self._fix_r2pipe()
            symbols = json.loads(payload)
            fcn_symb = [s for s in symbols if s['type'] == 'FUNC']
        except Exception as e:
            print("EXCEPTION 6: {}".format(e))
            fcn_symb = []
        return fcn_symb

    def analyze(self):
        if self.use_symbol:
            function_list = self.find_functions_by_symbols()
        else:
            function_list = self.find_functions()

        result = {}

        for my_function in function_list:
            if self.use_symbol:
                address = my_function['vaddr']
            else:
                address = my_function['offset']

            try:
                cfg, acfg, lstm_cfg = self.function_to_cfg(my_function)
                if cfg is not None:
                    result[my_function['name']] = {'cfg': cfg, "acfg": acfg, "lstm_cfg": lstm_cfg, "address": address}
            except:
                print("Error in functions: {} from {}".format(my_function['name'], self.filename))
                pass
        return result

    def close(self):
        self.r2.quit()

    def __exit__(self, exc_type, exc_value, traceback):
        self.r2.quit()
