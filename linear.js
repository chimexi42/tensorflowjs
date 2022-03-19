import * as tf from '@tensorflow/tfjs';


const xs = tf.tensor2d([1,2,3,4], [4,1])
const ys = tf.tensor2d([1,3,5,7], [4,1])

const model= tf.sequential()
model.add(tf.layers.dense({
    inputShape: [1],
    units:1
}))

model.compile({
    optimizer:'sgd',
    loss: 'meanSquaredError'
})

await model.fit(xs, ys, {epochs: 1000})
model.predict(tf.tensor2d([[5]], [1,1])).print()