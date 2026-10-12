#model structure by providing names that are packed with sequence of letters that will be used to then statistically predict
#next token in sequence: e.g., name 'Gemini' tells us 'G' is likely to be first letter; 'e' follows second and so on (for 50k names)
import torch
import matplotlib.pyplot as plt
#used to transform integer indexes into float vectors that neural net can actually process
import torch.nn.functional as F

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

#calculate sum of each row and normalise to add up to probability of sum of 1
#smoothing probability by +1 - that way avoiding inf when 0.0 probability for certain bigram
P = (N+1).float()
P /= P.sum(1, keepdims=True)
#manual seed for deterministic sequence
g = torch.Generator().manual_seed(6436364321)

# for i in range(20):
#     out = []
#     index = 0
#     while True:
#         #using fixed random seed generator and prob. of next char,
#         #we randomly choose one char which we sample and feed to determine next one in loop
#         p = P[index]
#         index = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         out.append(ceps[index])
#         if index == 0:
#             break
#     print(''.join(out))

#we want to ensure model assigns high probabilities to correct next char in sequence.
#we do this by repeatedly checking loss, calculating patterns and nudging weights in opposite direction
log_likelihood = 0.0
n = 0
for w in words:
    chrs = ['.'] + list(w) + ['.']
    for chr1, chr2 in zip(chrs, chrs[1:]):
        spec1 = spec[chr1]
        spec2 = spec[chr2]
        prob = P[spec1, spec2]
        #maximise likelihood of data in relation to probabilities of bigrams
        logprob= torch.log(prob)
        #eqv to maximising log likelihood (monotonic=one direction, !decrease/increase)
        #and equivalent of minimising both negative and average negative log likelihood
        log_likelihood += logprob
        n += 1
        #print(f'{chr1}{chr2}: {prob:.4f} {logprob:.4f}')

#since probabilities are between 0-1, when multiplying thousands of em we get infinitesimally
#small float numbers which aren't handled accurately -> take their logarithm means we need to simply perform additions (much stable/easier)
# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')


#training set of bigrams
inputs, targets = [], []

for w in words:
    #special characters to indicate where start/end
    chrs = ['.'] + list(w) + ['.']
    #pair neighbouring characters -> bigrams
    for chr1, chr2 in zip(chrs, chrs[1:]):
        spec1 = spec[chr1]
        spec2 = spec[chr2]
        inputs.append(spec1)
        targets.append(spec2)
#we convert inputs/targets into integers based on their index in alphabet
#then, converting into PyTorch tensors, that serve as actual dataset the neural network uses during training
#to learn how to predict next character
inputs = torch.tensor(inputs)
outputs = torch.tensor(targets)
num = inputs.nelement()
#print('number of examples: ', num)

#we initialise normal distribution and organise weights into 27x27 matrix so that we can compute
#firing rate of 27 input neurons across all inputs in parallel (efficiency in batches)
inputs_encoded = F.one_hot(inputs, num_classes=27).float()
W = torch.randn((27, 27), generator=g, requires_grad=True)

#training ground
for k in range(100):
    #transform into positive log counts that can actually be processed into probabilities
    log_counts = inputs_encoded @ W
    pos_counts = log_counts.exp()
    probs = pos_counts / pos_counts.sum(1, keepdims=True)

    #vectorised average negative log-likelihood (the loss) for 6 samples so its easier to debug
    #print(probs.shape) #[6, 27]
    loss = -probs[torch.arange(num), targets].log().mean()
    #backward pass - or, adjusting weights based of on actual calculated loss
    W.grad = None
    loss.backward()
    #updating tensor
    W.data += -50 * W.grad
    #print(loss.item())

#test sampling from model "neural net"
g = torch.Generator().manual_seed(6436364321)

for i in range(5):
    out = []
    index = 0
    while True:

        #before:
        #probs = P[index]
        #

        #now:
        inputs_encoded = F.one_hot(torch.tensor([index]), num_classes=27).float()
        log_counts = inputs_encoded @ W #predict log counts
        pos_counts = log_counts.exp() #counts = N
        probs = pos_counts / pos_counts.sum(1, keepdims=True) #probability for next char
        #

        index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(ceps[index])
        if index == 0:
            break
    print(''.join(out))


