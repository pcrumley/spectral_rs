import numpy as np

rng = np.random.default_rng(seed=42)
arr2 = rng.random((12, 24))
fft = np.fft.fft2(arr2)
input = []
out = []
for row in range(arr2.shape[0]):
    for col in range(arr2.shape[1]):
        input.append(arr2[row,col])
        out.append(fft[row, col])
print("let input: Vec<Float> = vec![", " ".join(f"Complex::new({elm}, 0.0)," for elm in input), '];')
print("let out: Vec<Float> = vec![", " ".join(f"Complex::new({elm.real}, {elm.imag})," for elm in out), '];')
# print("fft2 real = ", fft_re)
# print("fft2 imag = ", fft_im)
