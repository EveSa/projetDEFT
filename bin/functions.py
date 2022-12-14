import xml.etree.ElementTree as ET
from tqdm import tqdm
from yaspin import yaspin
from os import path


def file2data(lang, mode):
    '''
    train data extraction from files

    Parameters
    ----------
    lang : str
    mode : str
        among 'appr', 'test'

    Returns
    -------
    lst
        a list containing the data texts
    lst
        a list containing the target in the same order
    '''
    if mode=="appr":
        file_lang='files/deft09_parlement_appr_'+lang+'.xml'
    elif mode=="test":
        file_lang='files/deft09_parlement_test_'+lang+'.xml'
    else:
        raise ValueError("mode must be among 'test' or 'appr'")

    for i in tqdm(range(100), desc="get file_data"):
        with open(file_lang, 'r') as file:
            tree = ET.parse(file)
            root = tree.getroot()
            data=[]
            target=[]
            for doc in root.iter('doc'):
                text=[]
                for p in doc.iter('p'):
                    paragraph=p.text
                    if paragraph is not None:
                        paragraph=paragraph.replace(u'\xa0', u' ')
                        text.append(paragraph)
                if mode =="appr":
                    for eval in doc.iter('EVAL_PARTI'):
                        tag = eval.find('PARTI').attrib['valeur']
                    target.append(tag)
                data.append('. '.join(text))
        if mode =="test":
            ref_lang='files/deft09_parlement_ref_'+lang+'.txt'
            with open(ref_lang, 'r') as file:
                lines=file.readlines()
                target=[]
                line_to_remove=[]
                for i in range(len(lines)) :
                    if lines[i].split('\t')[1].strip('\n') == '':
                        line_to_remove.append(i)
                    else :
                        target.append(lines[i].split('\t')[1].strip('\n'))
            data = [datum for i, datum in enumerate(data) if i not in line_to_remove]
        
    with open('data/deft09_parlement_'+mode+'_'+lang+'_data.lst', 'w') as f:
        for datum in data :
            f.write(str(datum)+'\n')
    with open('data/deft09_parlement_'+mode+'_'+lang+'_target.lst', 'w') as f:
        for tag in target:
            f.write(str(tag)+'\n')

    return [datum.lower() for datum in data], [tag.lower() for tag in target]

def get_data(lang):
    '''
    get data per language from the file made by file2data

    Parameters
    ----------
    lang

    Returns
    -------
    data : list

    target : list

    test_data : list

    test_target : list
    '''
    filename='data/deft09_parlement_appr_'+lang+'_data.lst'
    if path.exists(filename):
        with yaspin(text="getting appr data") as sp:
            with open('data/deft09_parlement_appr_'+lang+'_data.lst', 'r') as f:
                data=f.read().lower().splitlines()
            with open('data/deft09_parlement_appr_'+lang+'_target.lst', 'r') as f:
                target=f.read().lower().splitlines()
            sp.ok('✔')
    else:
        data, target = file2data(lang, 'appr')

    filename='data/deft09_parlement_test_'+lang+'_data.lst'
    if path.exists(filename):
        with yaspin(text="getting test data") as sp:
            with open('data/deft09_parlement_test_'+lang+'_data.lst', 'r') as f:
                test_data=f.read().lower().splitlines()
            with open('data/deft09_parlement_test_'+lang+'_target.lst', 'r') as f:
                test_target=f.read().lower().splitlines()
            sp.ok('✔')
    else : 
        test_data, test_target = file2data(lang, 'test')
    
    return data, target, test_data, test_target