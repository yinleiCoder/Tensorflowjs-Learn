import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {getData} from './data.js';
// import {getData} from '../xor/data.js';

// 欠拟合与过拟合
// 欠拟合：数据复杂，模型简单，拟合不了
// 过拟合：模型太过复杂强大，拟合掉了数据

window.onload = async () => {
    // 加载带有噪音的二分类数据集
    const data = getData(200, 3);// 正太分布
    // const data = getData(200);// xor数据集
    // console.log(data);
    tfvis.render.scatterplot(
        {name: '训练数据'},
        {
            values: [
                data.filter(p=>p.label === 1),
                data.filter(p=>p.label === 0)
            ]
        }
    );
    // 使用简单神经网络演示欠拟合
    // 加载非线性的XOR数据集
    // 单个神经网络
    // const model = tf.sequential();
    // model.add(tf.layers.dense({
    //     units: 1,
    //     activation: 'sigmoid',
    //     inputShape: [2]
    // }));
    // model.compile({
    //     loss: tf.losses.logLoss,
    //     optimizer: tf.train.adam(0.1)
    // });
    // const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    // const labels = tf.tensor(data.map(p => p.label));
    // await model.fit(inputs, labels, {
    //     validationSplit: 0.2,
    //     epochs: 200,
    //     callbacks: tfvis.show.fitCallbacks(
    //         {name: '训练效果'},
    //         ['loss', 'val_loss'],
    //         {callbacks: ['onEpochEnd']}
    //     )
    // });
    // 使用复杂模型演示过拟合
    // 加载带有噪音的二分类数据集
    // 多层神经网络
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [2],
        activation: 'tanh',
        // 权重衰减法
        // kernelRegularizer: tf.regularizers.l2({ l2: 1}),
    }));
    // 丢弃法
    model.add(tf.layers.dropout({
        rate: 0.9 
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
    }));
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });
    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));
    await model.fit(inputs, labels, {
        validationSplit: 0.2,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练效果'},
            ['loss', 'val_loss'],
            {callbacks: ['onEpochEnd']}
        )
    });
    // 早停法

}