from numba import guvectorize, float64           
import numpy as np

@guvectorize([(float64[:],
             float64[:])], '(n)->(n)', nopython=True, target='parallel')
def binomial1DFilter(xArr, ans):
    i = 0
    ans[0] = xArr[-1]+2*xArr[0]+xArr[1]
    i += 1
    while i < xArr.shape[0]-1:
        ans[i] = xArr[i - 1] + 2 * xArr[i] + xArr[i + 1]
        i += 1
    ans[i] = xArr[i - 1] + 2 * xArr[i] + xArr[0]


def wrappedFilter(xArr, NPass):
    for i in range(NPass):
        xArr = binomial1DFilter(xArr)
        xArr = binomial1DFilter(xArr.T).T
    xArr *= 16**-NPass
    return xArr


input = np.random.rand(12, 24)
arr_in = [input[-1, 23]]
for elm in input[-1, :]:
    arr_in.append(elm)
arr_in.append(input[-1, 0])

for row in range(input.shape[0]):
    arr_in.append(input[row, -1])
    for col in range(input.shape[1]):
        arr_in.append(input[row, col])
    arr_in.append(input[row, 0])

arr_in.append(input[0, -1])
for elm in input[0, :]:
    arr_in.append(elm)
arr_in.append(input[0, 0])


input = wrappedFilter(input, 4)

arr_out = [input[-1, -1]]
for elm in input[-1, :]:
    arr_out.append(elm)
arr_out.append(input[-1, 0])

for row in range(input.shape[0]):
    arr_out.append(input[row,-1])
    for col in range(input.shape[1]):
        arr_out.append(input[row, col])
    arr_out.append(input[row, 0])

arr_out.append(input[0, -1])
for elm in input[0, :]:
    arr_out.append(elm)
arr_out.append(input[0, 0])


print("fld.spatial = vec![", " ".join(f"{elm}," for elm in arr_in), '];')
print("expected_out.spatial = vec![", " ".join(f"{elm}," for elm in arr_out), '];')

