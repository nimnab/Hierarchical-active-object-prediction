import numpy as np
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
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

        self.noneremove()
        self.reverse_hierarchy = self.inverse_hierachy()
        self.encoddict, self.revencoddict = self.create_encoddic()
        self.level1, self.level2, self.level3 = self.outputindexes()
        self.inputs, self.labels = self.datagen()

    def generate_fold(self, seed):
        self.train, self.test = train_test_split(self.biglist, test_size=0.2, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            self.inputs,
            self.labels,
            test_size=0.2,
            random_state=seed,
        )
        self.dtrain = Mydata(X_train, y_train)
        self.dtest = Mydata(X_test, y_test)

    def outputindexes(self):
        level1 = []
        level2 = []
        level3 = []
        for node in self.reverse_hierarchy:
            if len(self.reverse_hierarchy[node])==1:
                level1.append(self.encoddict[node])
            elif len(self.reverse_hierarchy[node])==2:
                level2.append(self.encode(node)[0])
            else:
                level3.append(self.encode(node)[0])
        return level1, level2, level3

    def datagen(self):
        inputdim = len(self.encoddict)
        lens = []
        for lis in self.biglist:
            lens.append(sum([len(j) for j in lis]))
        maxlen = max(lens) + 2
        encodedinputs = np.empty((0, maxlen, inputdim))
        encodedlabels = np.empty((0, maxlen, inputdim))
        for manual in self.biglist:
            xvectors = np.zeros((maxlen, inputdim))
            yvectors = np.zeros((maxlen, inputdim))
            xvectors[0, self.encoddict['START']] = 1
            ind = 0
            for step in manual:
                if step:
                    for tool in step:
                        xvectors[ind + 1, self.encode(tool)] = 1
                        yvectors[ind] = xvectors[ind + 1]
                        ind += 1
            yvectors[ind, self.encoddict['END']] = 1
            encodedinputs = np.append(encodedinputs, [xvectors], axis=0)
            encodedlabels = np.append(encodedlabels, [yvectors], axis=0)
        return encodedinputs, encodedlabels

    def encode(self, obj):
        indexes = []
        if obj in self.reverse_hierarchy:
            parent = ''
            while parent != '<ROOT>':
                child, parent = self.seprateparent(obj)
                indexes.append(self.encoddict[child])
                obj = parent
        else:
            indexes.append(self.encoddict['UNK'])
        return indexes

    def create_encoddic(self):
        encoddict = dict()
        encoddict['START'] = 0
        index = 1
        for obj in self.reverse_hierarchy:
            obj = self.seprateparent(obj)[0]
            if obj not in encoddict:
                encoddict[obj] = index
                index += 1
        encoddict['END'] = index
        encoddict['UNK'] = index + 1
        revencoddict = {v: k for k, v in encoddict.items()}
        return encoddict, revencoddict

    def seprateparent(self, node):
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
        return revrese_hirachy

    def isleaf(self, node):
        if node in self.class_hierarchy.keys():
            return False
        else:
            return True

    def noneremove(self):
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
