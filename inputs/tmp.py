# %%
from Reps import Reps
import nltk
from copy import deepcopy as cp
cmudict = nltk.corpus.cmudict.dict()

pool = ['the', 'and', 'but', 'this']
r = Reps(pool, outliers=['and'], onehot=True, terminals=True)


#%%
pad = 9
onehot = True
terminals = True
phonforms = r.phonforms
phonreps = r.phonreps
orthreps = r.orthreps
cmudict = r.cmudict

# create the pad by assigning to '_'
if pad != 0:
    padrep = []
    for f in phonreps['_']:
        padrep.append(pad)
    phonreps['_'] = padrep


repd = r.phonforms['the']
maxlen = 4
veclengths = set([len(v) for v in phonreps.values()])
assert(len(veclengths) == 1), 'Phonological feature vectors across phonreps have different lengths.'
phonemeComplexity = next(iter(veclengths))

if not onehot:
    veclengths = set([len(v) for v in orthreps.values()])
    assert(len(veclengths) == 1), 'Orthographic feature vectors across phonreps have different lengths.'

if terminals:
    cmudictSOS = {}
    cmudictEOS = {}
    for word, phonform in cmudict.items():
        sos = cp(phonform)
        eos = cp(phonform)

        sos.insert(0, '#')
        eos.append('%')

        cmudictSOS[word] = sos
        cmudictEOS[word] = eos




#%%
sos = {}

for word in pool:
    padlen = maxlen-len(phonforms[word])
    p = cp(phonreps['#'])
    p.append(phonize(cmudict([word]))
    for slot in range(padlen):
        p.append(phonreps['_'])
    phon_sos_masked_right[word] = p