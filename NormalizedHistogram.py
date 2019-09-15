import matplotlib.pyplot as pyplot
t=[1,3,4,2,11,18,3,4,5,6,8,9,0,5,5]
n = float(len(t))
hist = {}
for x in t:
    hist[x] = hist.get(x, 0) + 1

print(hist)
vals, freqs=[],[]

for x, freq in hist.items():
    vals.append(x)
    freqs.append(freq)

pyplot.bar(vals, freqs)
pyplot.show()

pmf = {}
for x, freq in hist.items():
    pmf[x] = freq / n
print(pmf)

vals, freqs=[],[]
for x, freq in pmf.items():
    vals.append(x)
    freqs.append(freq)

pyplot.bar(vals, freqs)
pyplot.show()

