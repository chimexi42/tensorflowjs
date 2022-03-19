import * as tf from '@tensorflow/tfjs';
import iris from "./iris.json"
import irisTesting from "./irisTesting.json"


const trainingData = tf.tensor2d(iris.map(item =>{
    item.sepalLength, item.sepalWidth, item.petalLength, item.petalWidth
}))
const outputData = tf.tensor2d(iris.map(item =>{
   item.species === "setosa"? 1: 0,
   item.species === "virginica"? 1: 0,
   item.species === "Versicolor"? 1: 0
}))

const testingData = tf.tensor2d(irisTesting.map(item =>{
   item.sepalLength, item.sepalWidth, item.petalLength, item.petalWidth
}))

const model = tf.sequential()
model.add(tf.layers.dense({ 
    inputShape:[4],
    activation: "sigmoid", 
     units:5
}))

model.add({
    inputShape:[5],
    activation:"sigmoid",
    units:3
})
model.add({
    activation:"sigmoid",
    units:3
})

model.compile({
    loss: "meanSquaredError",
    optimizer:tf.train.adam(.06)
})
const startTime = Date.now()
await model.fit(trainingData, outputData, {epochs:100})
.then((history)=> {
    console.log("DONE!", Date.now- startTime)
    console.log(history)
    model.predict(testingData).print()
})

