import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import { getData } from './data.js';
import { input } from '@tensorflow/tfjs';

window.onload = async () => {
    // 逻辑回归

    // 加载二分类数据
    const data = getData(400); // 这里是自定义的函数来模拟大数据，只是为了学习
    //console.log(data);
    // 散点图
    tfvis.render.scatterplot(
        { name: '逻辑回归训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );
    // 定义模型结构：带有激活函数的单个神经元
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation:'sigmoid'// sigmoid将输出值压缩到0~1
    }));
    // 损失函数： 对数损失logLoss
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1) // 这个优化器可以帮我们自动调整参数
    });
    // 训练模型并可视化训练过程
    const inputs = tf.tensor(data.map(p=> [p.x, p.y]));
    const labels = tf.tensor(data.map(p=> p.label));
    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    });
    // 进行预测
    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value *1]]));
        alert(`预测结果： ${pred.dataSync()[0]}`);
    };
}