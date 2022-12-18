from joblib import dump, load
from os import path
from statistics import mean
import matplotlib.pyplot as plt

from yaspin import yaspin

from adapt_data import*

class Model:
    '''Tous les modèles prennent comme paramètres :

    Entrées
    -------
    name : le nom de la sauvegarde
    lang : la langue d'entraînement du modèle
    X_train : les données d'entraînement
    y_train : les targets d'entraînement
    X_test : les données de test
    
    Sortie
    ------
    y_pred : les targets prédits pour les X_test
    classes : les classes utilisées

    Tous les imports sont faits à l'intérieur des fonctions pour ne pas surcharger l'appel de ce module.
    Toutes les fonctions sont définies selon le même principe :
        on vérifie d'abord que le modèle n'a pas déjà été entraîné
        on entraîne un nouveau modèle si besoin
    '''

    def __init__(self, model_name, name, lang, X_train, y_train, X_test, max_depth=None):
        self.model_name = model_name
        self.name = name
        self.lang = lang
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.max_depth = max_depth
    
    def __call__(self, model_name, name, lang, X_train, y_train, X_test, max_depth=None):
        if model_name == 'decision_tree' :
            return self.decision_tree(name, lang, X_train, y_train, X_test, max_depth)
        elif model_name == 'random_forest' :
            return self.random_forest(name, lang, X_train, y_train, X_test, max_depth)
        elif model_name == 'logistic_regression' :
            return self.logistic_regression(name, lang, X_train, y_train, X_test)
        elif model_name == 'linearsvc_classification' :
            return self.linearsvc_classification(self, name, lang, X_train, y_train, X_test)
        elif model == 'svc_classification' :
            return self.svc_classification(self, name, lang, X_train, y_train, X_test)
        else :
            return self.SGD_classification(self, name, lang, X_train, y_train, X_test)

    def decision_tree(self, name, lang, X_train, y_train, X_test, max_depth=None):
        from sklearn import tree
        '''
        Decision tree training
        '''
        filename = 'model/DecisionTree/'+ name +'_' + lang +'.joblib'

        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf = load(filename)
        else :
            with yaspin(text="training Tree") as sp:
                clf = tree.DecisionTreeClassifier(max_depth=max_depth)
                clf = clf.fit(X_train, y_train)
                dump(clf, filename)
                sp.ok('✔')

        y_pred=clf.predict(X_test)
        classes = clf.classes_
        print("tree depth : " + str(clf.tree_.max_depth))

        #fig = plt.figure(figsize=(25,20))
        #_ = tree.plot_tree(clf)
        #fig.savefig("decistion_tree.png")

        return y_pred, classes

    def random_forest(self, name, lang, X_train, y_train, X_test, max_depth=None):
        from sklearn.ensemble import RandomForestClassifier
        '''
        random forest training
        '''

        filename = 'model/RandomForest/'+name+'_' + lang +'.joblib'
        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf = load(filename)
        else :
            with yaspin(text="training randomForest") as sp:
                clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
                clf.fit(X_train, y_train)
                dump(clf, filename)
                sp.ok('✔')

        y_pred=clf.predict(X_test)
        classes = clf.classes_
        depth = [estimator.tree_.max_depth for estimator in clf.estimators_]
        print("trees depths : "+str(depth))
        print("mean depth : "+str(mean(depth)))

        return y_pred, classes

    def logistic_regression(self, name, lang, X_train, y_train, X_test):
        '''
        logistic regression training 
        '''
        from sklearn.linear_model import LogisticRegression

        filename = 'model/LogisticRegression/'+name+'_' + lang +'.joblib'
        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf = load(filename)
        else :
            with yaspin(text="training logisticRegression") as sp:
                clf = LogisticRegression(random_state=0)
                clf.fit(X_train, y_train)
                dump(clf, filename)
                sp.ok('✔')

        y_pred = clf.predict(X_test)
        classes = clf.classes_
        print("nb of iterations : " + str(clf.n_iter_))

        return y_pred, classes

    def linearsvc_classification(self, name, lang, X_train, y_train, X_test):
        '''
        svc classifiation training 
        '''
        from sklearn.svm import SVC, LinearSVC
        from sklearn.model_selection import GridSearchCV
        import pandas as pd

        filename = 'model/LinearSVC/'+name+'_' + lang +'.joblib'
        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf_best = load(filename)
        else :
            with yaspin(text="training SVCClassifier") as sp:
                clf = LinearSVC(C=1.25)
                clf.fit(X_train, y_train)
                dump(clf, filename)
                sp.ok('✔')

        y_pred=clf.predict(X_test)
        classes = clf.classes_
        print("nb of iterations : " + str(clf_best.n_iter_))

    def svc_classification(self, name, lang, X_train, y_train, X_test):
        '''
        svc classifiation training 
        '''
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        import pandas as pd

        filename = 'model/SVCClassification/'+name+'_' + lang +'.joblib'

        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf_best = load(filename)
        else :
            with yaspin(text="training SVCClassifier") as sp:
                parameters = {'kernel':['poly']}
                svc = SVC()
                clf = GridSearchCV(svc, parameters)
                clf.fit(X_train, y_train)
                clf_best=clf.best_estimator_
                dump(clf_best, filename)
                sp.ok('✔')
                print("best params : "+str(clf.best_params_))
                print("nb of features : " +str(clf.n_features_in_))
                df = pd.DataFrame(clf.cv_results_)

                from IPython.display import display
                display(df)

        y_pred=clf_best.predict(X_test)
        classes = clf_best.classes_
        print("nb of iterations : " + str(clf_best.n_iter_))

        return y_pred, classes

    def SGD_classification(self, name, lang, X_train, y_train, X_test):
        '''train SGD'''
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import GridSearchCV

        filename = 'model/SGDClassifier/'+name+'_' + lang +'.joblib'
        if path.exists(filename):
            print("✔ using pre dumped model ... ")
            clf_best = load(filename)
        else :
            with yaspin(text="training SGD") as sp:
                parameters = {
                    'loss':['modified_huber'], 
                    'class_weight':[{'eldr':7.5, 'gue-ngl':5, 'ppe-de':2.5, 'pse':2, 'verts-ale':5}], 
                    'epsilon':[0.3 ], 
                    'warm_start':[True]
                    }

                scoring='f1_macro'
                sdg = SGDClassifier()
                clf = GridSearchCV(sdg, parameters, scoring=scoring)
                clf.fit(X_train, y_train)
                clf_best=clf.best_estimator_
                dump(clf_best, filename)
                sp.ok('✔')
                print("best params : "+str(clf.best_params_))
                print("nb of features : " +str(clf.n_features_in_))
                df = pd.DataFrame(clf.cv_results_)

                from IPython.display import display
                display(df)

        y_pred=clf_best.predict(X_test)
        classes = clf_best.classes_
        print("nb of iterations : " + str(clf_best.n_iter_))

        return y_pred, classes

    def bert_classifn(self, name, lang, X_train, y_train, X_test):
        '''
        Essai d'entraînement avec BERT
        Tentative avortée
        '''
        import torch
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


        tokenizer = CamembertTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model = CamembertForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")


        possible_labels= [
            'Verts-ALE',
            'PPE-DE',
            'PSE',
            'ELDR',
            'GUE-NGL'
            ]
        label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = ORTModelForSequenceClassification.from_pretrained('bert-base-uncased')
        encoded_data_train = tokenizer.batch_encode_plus(
            X_train,
            add_special_tokens=True,
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=256, 
            return_tensors='pt'
        )

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(df[df.data_type=='train'].label.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(df[df.data_type=='val'].label.values)

        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

        batch_size = 3

        dataloader_train = DataLoader(dataset_train, 
                                      sampler=RandomSampler(dataset_train), 
                                      batch_size=batch_size)

        dataloader_validation = DataLoader(dataset_val, 
                                           sampler=SequentialSampler(dataset_val), 
                                           batch_size=batch_size)

        model.to(device)

        model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))

        _, predictions, true_vals = evaluate(dataloader_validation)

        y_pred=clf.best_estimator_.predict(X_test)
        classes = clf.classes_
        print("nb of iterations : " + str(clf.n_iter_))

        return y_pred, classes