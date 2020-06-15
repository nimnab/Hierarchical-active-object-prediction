import numpy as np
from sklearn.model_selection import train_test_split

tools = '/home/nnabizad/code/hierarchical/data/mac/mac_tools'
# tools = '/home/nnabizad/code/toolpred/data/yam/yam_tools'
objects = '/home/nnabizad/code/hierarchical/data/mac/mac_parts'
# objects = '/home/nnabizad/code/toolpred/data/yam/yam_ings'

min_freq = 1
# glove_embedding = WordEmbeddings('glove')
# glove_embedding = WordEmbeddings('/hri/localdisk/nnabizad/w2v/glove100_word2vec1')
# glovedim = 100
# document_embeddings = DocumentPoolEmbeddings([glove_embedding],
#                                              pooling='max')
unknowns = []


class Mydata():
    def __init__(self, dat, tar, titles=None):
        self.input = dat
        self.target = tar
        self.titles = titles


class Data:
    def __init__(self, obj='mac_tools'):

        if obj.endswith('tools'):
            self.biglist = np.load(tools + '.pkl', allow_pickle=True)
            self.class_hierarchy = np.load(tools + '_hi.pkl', allow_pickle=True)
        else:
            self.biglist = np.load(objects + '.pkl', allow_pickle=True)
            self.class_hierarchy = np.load(objects + '_hi.pkl', allow_pickle=True)

        self.none_remove()
        self.reverse_hierarchy = self.inverse_hierachy()

        self.encoddict_hir, self.decoddict_hir = self.create_hierarchical_encoddic()
        self.level1, self.level2, self.level3 = self.hierarchy_indices()
        self.encoddict_flat, self.decoddict_flat = self.create_flat_encoddic()
        self.inputs_hir, self.labels_hir = self.data_gen(hierarchical=True)
        self.inputs_flat, self.labels_flat = self.data_gen(hierarchical=False)

    def generate_fold(self, seed, hierarchical):
        self.train, self.test = train_test_split(self.biglist, test_size=0.2, random_state=seed)
        if hierarchical:
            inp, outp = self.inputs_hir, self.labels_hir
        else:
            inp, outp = self.inputs_flat, self.labels_flat
        X_train, X_test, y_train, y_test = train_test_split(
            inp,
            outp,
            test_size=0.2,
            random_state=seed,
        )
        self.dtrain = Mydata(X_train, y_train)
        self.dtest = Mydata(X_test, y_test)

    def hierarchy_indices(self):
        level1 = []
        level2 = []
        level3 = []
        for node in self.reverse_hierarchy:
            if len(self.reverse_hierarchy[node]) == 1:
                level1.append(self.encoddict_hir[node])
            elif len(self.reverse_hierarchy[node]) == 2 and node in self.class_hierarchy:
                print(node)
                level2.append(self.encode(node)[0])
            else:
                level3.append(self.encode(node)[0])
        return level1, level2, level3

    def data_gen(self, hierarchical):
        encoddict = self.encoddict_hir if hierarchical else self.encoddict_flat
        inputdim = len(encoddict)
        lens = []
        for lis in self.biglist:
            lens.append(sum([len(j) for j in lis]))
        maxlen = max(lens) + 2
        encodedinputs = np.empty((0, maxlen, inputdim))
        encodedlabels = np.empty((0, maxlen, inputdim))
        for manual in self.biglist:
            xvectors = np.zeros((maxlen, inputdim))
            yvectors = np.zeros((maxlen, inputdim))
            xvectors[0, encoddict['START']] = 1
            ind = 0
            for step in manual:
                if step:
                    for tool in step:
                        xvectors[ind + 1, self.encode(tool, hierarchical)] = 1
                        yvectors[ind] = xvectors[ind + 1]
                        ind += 1
            yvectors[ind, encoddict['END']] = 1
            encodedinputs = np.append(encodedinputs, [xvectors], axis=0)
            encodedlabels = np.append(encodedlabels, [yvectors], axis=0)
        return encodedinputs, encodedlabels

    def encode(self, obj, hierarchical=True):
        indexes =  []
        obj = obj.strip()
        if obj in self.reverse_hierarchy:
            if hierarchical:
                parent = ''
                while parent != '<ROOT>':
                    child, parent = self.seprate_parent(obj)
                    indexes.append(self.encoddict_hir[child])
                    obj = parent
            else:
                indexes.append(self.encoddict_flat[obj])
        else:
            if hierarchical:
                objname = obj.split()
                for i in range(len(objname)-1,0,-1):
                    _obj = ' '.join(objname[-i:])
                    print(_obj)
                    if _obj in self.reverse_hierarchy:
                        indexes.append(self.encoddict_hir['UNK'])
                        parent = ''
                        while parent != '<ROOT>':
                            child, parent = self.seprate_parent(_obj)
                            indexes.append(self.encoddict_hir[child])
                            _obj = parent
                        break
                if not indexes:
                    indexes = [self.encoddict_hir['UNK']]
                    print('Unknown:', obj)
            else:
                indexes= self.encoddict_flat['UNK']
                print('Unknown:', obj)
        return indexes

    def create_hierarchical_encoddic(self):
        encoddict = dict()
        encoddict['START'] = 0
        index = 1
        for obj in self.reverse_hierarchy:
            obj = self.seprate_parent(obj)[0]
            if obj not in encoddict:
                encoddict[obj] = index
                index += 1
        # encoddict['END'] = index
        # encoddict['UNK'] = index + 1
        decoddict = {v: k for k, v in encoddict.items()}
        return encoddict, decoddict

    def create_flat_encoddic(self):
        objs = set([i.strip() for j in self.biglist for k in j for i in k])
        encoddict = dict()
        encoddict['START'] = 0
        index = 1
        for obj in objs:
            if obj in self.reverse_hierarchy and obj not in encoddict:
                encoddict[obj] = index
                index += 1
        encoddict['END'] = index
        encoddict['UNK'] = index + 1
        decoddict = {v: k for k, v in encoddict.items()}
        return encoddict, decoddict

    def seprate_parent(self, node):
        if node in self.reverse_hierarchy:
            return node.replace(self.reverse_hierarchy[node][0], '').strip(), self.reverse_hierarchy[node][0]

    def inverse_hierachy(self):
        # collect samples for non leafs
        nonleafs = {tool for tool in self.class_hierarchy['<ROOT>'] if not self.isleaf(tool)}
        revrese_hirachy = dict()
        for tool in self.class_hierarchy['<ROOT>']: revrese_hirachy[tool] = ['<ROOT>']
        for parent in nonleafs:
            for child in self.class_hierarchy[parent]:
                revrese_hirachy[child] = [parent, '<ROOT>']
                if not self.isleaf(child):
                    for grandchild in self.class_hierarchy[child]:
                        if self.isleaf(grandchild):
                            revrese_hirachy[grandchild] = [child, parent, '<ROOT>']
        revrese_hirachy['START'] = ['<ROOT>']
        revrese_hirachy['END'] = ['<ROOT>']
        revrese_hirachy['UNK'] = ['<ROOT>']
        return revrese_hirachy

    def isleaf(self, node, noparent=False):
        keys = list(self.class_hierarchy.keys())
        keys.remove('<ROOT>')
        if noparent: keys = [self.seprate_parent(i)[0] for i in keys]
        if not node in keys:
            return True
        else:
            return False

    def none_remove(self):
        for key in self.class_hierarchy:
            for elem in self.class_hierarchy[key]:
                if elem.startswith('None'):
                    self.class_hierarchy[key].remove(elem)


if __name__ == '__main__':
    data = 'mac_tools'
    # data = 'mac_parts'
    mydata = Data(obj=data)
    print()
    # t = Topicmodel(0)
