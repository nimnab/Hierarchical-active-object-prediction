import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class Hierarchy():
    def __init__(self,
                 data = None):
        self.hierarchy = data.class_hierarchy
        self.data = data

    def testclf(self,clf):
        clf.fit(self.rootx, self.rooty)
        preds = clf.predict(self.rootxtest)
        acc = accuracy_score(self.rootytest, preds)
        print(acc, flush=True)
        return acc


    def fit(self, xtrain, ytrain, xtest):
        rootleafs, rootnonleafs, rootreverse = self.node_data('<ROOT>')
        rootx, rooty = self.rolled_data(rootleafs, rootreverse, xtrain, ytrain)
        rclf = clone(self.clf)
        rclf.fit(rootx, rooty)
        self.clfs['<ROOT>'] = rclf
        for nleaf in rootnonleafs:
            leafs, nonleafs, revh = self.node_data(nleaf)
            _x, _y = self.rolled_data(leafs, revh, xtrain, ytrain)
            if _x:
                print('number of samples for {} is {}'.format(nleaf,len(_y)))
                _clf = clone(self.clf)
                _clf.fit(_x, _y)
                self.clfs[nleaf] = _clf
            for _nleaf in nonleafs:
                _leafs, _nonleafs, _revh = self.node_data(_nleaf)
                _xt, _yt = self.rolled_data(_leafs, _revh, xtrain, ytrain)
                if _xt:
                    print('number of samples for {} is {}'.format(_nleaf, len(_yt)))
                    _clft = clone(self.clf)
                    _clft.fit(_xt, _yt)
                    self.clfs[_nleaf] = _clft
        return 0

    def rolled_data(self, leafs, revrese_hirachy, xtrain, ytrain):
        x = []
        y = []
        for j in range(len(ytrain)):
            # for j in ytrain[i]:
            if ytrain[j] in leafs:
                y.append(ytrain[j])
                x.append(xtrain[j])
            elif ytrain[j] in revrese_hirachy:
                y.append(revrese_hirachy[ytrain[j]])
                x.append(xtrain[j])
        return x, y

    def node_data(self, node):
        # collect samples for non leafs
        leafs = {tool for tool in self.hierarchy[node] if self.isleaf(tool)}
        nonleafs = {tool for tool in self.hierarchy[node] if not self.isleaf(tool)}
        revrese_hirachy = dict()
        for parent in nonleafs:
            for child in self.hierarchy[parent]:
                if self.isleaf(child):
                    revrese_hirachy[child] = parent
                else:
                    for grandchild in self.hierarchy[child]:
                        if self.isleaf(grandchild):
                            revrese_hirachy[grandchild] = parent
                        else:
                            print(grandchild)
        return leafs, nonleafs, revrese_hirachy

    def encode(self, object):
        # if object in self.data.toolencoddic:
        #     return self.data.toolencoddic[object]
        # else:
            return object

    def siblings(self, inverse_hierachy, obj):
        sibling = [i for i in inverse_hierachy if inverse_hierachy[i] == inverse_hierachy[obj] and str(i).isdigit()]
        if obj in sibling: sibling.remove(obj)
        return sibling


    def onehot(self, ml_y, onhoty, siblings, mainprob):
        for ind in range(len(ml_y)):
            obj = ml_y[ind]
            if obj ==0:
                return onhoty
            else:
                if obj in siblings:
                    lis = [0]*len(self.data.toolencoddic)
                    siblen = len(siblings[obj])
                    sibprob = (1-mainprob)/siblen
                    for sib in siblings[obj]:
                        lis[sib] = sibprob
                    lis[obj]=mainprob
                    onhoty[ind] = lis
        return onhoty

    def inverse_hierachy(self, node, level):
        # collect samples for non leafs
        nonleafs = {tool for tool in self.hierarchy[node] if not self.isleaf(tool)}
        revrese_hirachy = dict()
        for tool in self.hierarchy['<ROOT>']: revrese_hirachy[self.encode(tool)] = tool
        for parent in nonleafs:
            for child in self.hierarchy[parent]:
                if self.isleaf(child):
                    revrese_hirachy[self.encode(child)] = parent
                else:
                    revrese_hirachy[self.encode(child)] = parent
                    for grandchild in self.hierarchy[child]:
                        if self.isleaf(grandchild):
                            if level == 1:
                                revrese_hirachy[self.encode(grandchild)] = parent
                            elif level == 2:
                                revrese_hirachy[self.encode(grandchild)] = child
                        else:
                            print(grandchild)
        for i in self.data.invtooldic:
            if i not in revrese_hirachy:
                revrese_hirachy[i] = i
        return revrese_hirachy

    def isleaf(self, node):
        if node in self.hierarchy.keys():
            return False
        else:
            return True

    def predict(self, xtest):
        ys = []
        for i, x in enumerate(xtest):
            label = self.clfs['<ROOT>'].predict(x.reshape(1, -1))[0]
            if self.isleaf(label):
                ys.append(label)
                # if len(step) >= lens[i]: break
            elif label in self.clfs:
                _y = self.clfs[label].predict(x.reshape(1, -1))[0]
                if self.isleaf(_y):
                    ys.append(_y)
                    # if len(step) >= lens[i]: break
                elif _y in self.clfs:
                    _yt = self.clfs[_y].predict(x.reshape(1, -1))[0]
                    if self.isleaf(_yt):
                        ys.append(_yt)
                        # if len(step) >= lens[i]: break
        return ys

    def predict_beem(self, xtest, lens):
        ys = []
        for i, x in enumerate(xtest):
            y = self.clfs['<ROOT>'].predict_proba(x.reshape(1, -1))[0]
            args = np.argsort(-y)
            labels = {self.clfs['<ROOT>'].classes_[args[c]]: y[args[c]] for c in range(lens[i])}
            for label in labels.copy():
                if not self.isleaf(label):
                    if label in self.clfs:
                        y1 = self.clfs[label].predict_proba(x.reshape(1, -1))[0]
                        args1 = np.argsort(-y1)
                        labs = {self.clfs[label].classes_[args1[c]]: y1[args1[c]] * labels[label] for c in
                                range(min(lens[i],len(args1)-1))}
                        del labels[label]
                        labels.update(labs)
                        for lab in labs:
                            if not self.isleaf(lab):
                                del labels[lab]
                                if lab in self.clfs:
                                    y2 = self.clfs[lab].predict_proba(x.reshape(1, -1))[0]
                                    args2 = np.argsort(-y2)
                                    labs3 = {self.clfs[lab].classes_[args2[c]]: y2[args2[c]] * labs[lab] for c in
                                             range(min(lens[i],len(args2)-1))}
                                    labels.update(labs3)

            sorted_labels = [j[0] for j in sorted(labels.items(), key=lambda kv: kv[1], reverse=True)][:lens[i]]
            ys.append(sorted_labels)
        return ys

    def accuracy (self, preds, targets , level=3):
        targets = targets.copy()
        if level < 3:
            rootreverse = self.inverse_hierachy('<ROOT>', level)
            for s in range(len(targets)):
                targets = [rootreverse[i] for i in targets if i in rootreverse]
                preds = [rootreverse[i] for i in preds if i in rootreverse]
        corrects = 0
        for i in range(len(targets)):
            if targets[i]==preds[i]:
                corrects +=1
            else:
                print(targets[i], preds[i])
        return corrects/len(targets)

