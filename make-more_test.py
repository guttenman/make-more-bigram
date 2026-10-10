#model structure by providing names that are packed with sequence of letters that will be used to then statistically predict
#next token in sequence: e.g., name 'Gemini' tells us 'G' is likely to be first letter; 'e' follows second and so on (for 50k names)
import torch
import matplotlib.pyplot as plt

#-> .lower due to how laggy graph becomes with capital letters
words = open('names.txt', 'r').read().lower().splitlines()

#count bigrams
chars = sorted(list(set(''.join(words))))
spec = {s:i+1 for i,s in enumerate(chars)}
spec['.'] = 0
# Reverse dictionary to use character labels in the graph
ceps = {i: s for s, i in spec.items()}
#52 chars (lower + capital) + 2 special chars = 54x54 matrix
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    #special characters to indicate where start/end
    chrs = ['.'] + list(w) + ['.']
    #pair neighbouring characters -> bigrams
    for chr1, chr2 in zip(chrs, chrs[1:]):
        spec1 = spec[chr1]
        spec2 = spec[chr2]
        N[spec1, spec2] += 1

# VERSION 1 Graph - Detailed, Laggy when zoom
plt.figure(figsize=(20, 20))
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = ceps[i] + ceps[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=8)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray", fontsize=7)
plt.axis('off')
plt.tight_layout()
# plt.show()

#VERSION 2 Graph - Simpler, Not laggy when zoom
# plt.figure(figsize=(14, 14))
# plt.imshow(N, cmap='Blues')

# plt.xticks(range(53), [ceps[i] for i in range(53)], rotation=90)
# plt.yticks(range(53), [ceps[i] for i in range(53)])

# plt.xlabel('Character that follows')
# plt.ylabel('Current character')
# plt.title('Character bigram frequencies')
# plt.show()

g = torch.Generator().manual_seed(6436364321)

for i in range(20):
    out = []
    index = 0
    while True:
        p = N[index].float()
        p = p/p.sum()
        index = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(ceps[index])
        if index == 0:
            break
    print(''.join(out))
