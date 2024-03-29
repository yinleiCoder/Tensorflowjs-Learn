import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import {getIrisData, IRIS_CLASSES} from './data.js';

// softmax: 概率和 =1
// iris数据集包含数据集和验证集
// 多分类
window.onload = async () => {
    // 加载iris数据集
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);// 15%的数据作为验证集
    // xTrain.print();
    // yTrain.print();
    // xTest.print();
    // yTest.print();
    // console.log(IRIS_CLASSES); // 中文分类

    // 定义模型结构：带有softmax的多层神经网络(非线性就需要多层神经网络)
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'
    }));
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));
    // 训练模型：交叉熵损失函数与准确度度量
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']
    });
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练效果'},
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {callbacks: ['onEpochEnd']}
        )
    });
    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};