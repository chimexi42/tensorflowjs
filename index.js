import * as tf from '@tensorflow/tfjs';

// const xs = tf.tensor2d([1, 2, 3, 4], [1, 4]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
// console.log(xs.print())
// console.log(ys.print())


const data = tf.tensor([0,0, 127.5, 255, 100, 50, 24, 55], [4,2], "int32")
const data2 = tf.tensor([0,0, 127, 255, 100, 50, 24, 55], [2,2,2])

data.print()
console.log(data.toString())

data2.print()


const values = []
for (let i =0; i<30; i++){
    values[i] = Math.floor((Math.random() * 101))
   
}
const shape = [2, 5, 3]

const data3 = tf.tensor3d(values, shape, "int32")
data3.print()

// const num = tf.scalar(3).print()
// data3.data().then(data=> console.log(data))

console.log(data3.dataSync())

const newVar = tf.variable(data3)
console.log(newVar)

const a = tf.tensor3d(values, shape, "int32")
const b = tf.tensor3d(values, shape, "int32")

const c = a.add(b)

c.print()

const first = tf.tensor2d([1,2], [1,2])
const second = tf.tensor2d([1,2,3,4], [2,2])

first.matMul(second).print()