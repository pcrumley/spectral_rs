import numpy as np

rng = np.random.default_rng(seed=42)
arr1 = rng.random(24)
fft = np.fft.fft(arr1)
print("let input: Vec<Complex<Float>> = vec![", " ".join(f"Complex::new({elm}, 0.0)," for elm in arr1), '];')
print("let out: Vec<Complex<Float>> = vec![", " ".join(f"Complex::new({elm.real}, {elm.imag})," for elm in fft), '];')

arr2 = rng.random((12, 24))
fft = np.fft.fft2(arr2)
input = []
out = []
for row in range(arr2.shape[0]):
    for col in range(arr2.shape[1]):
        input.append(arr2[row, col])
        out.append(fft[row, col])
# print("let input: Vec<Complex<Float>> = vec![", " ".join(f"Complex::new({elm}, 0.0)," for elm in input), '];')
# print("let out: Vec<Complex<Float>> = vec![", " ".join(f"Complex::new({elm.real}, {elm.imag})," for elm in out), '];')
