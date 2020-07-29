// 线性回归

import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { outerProduct } from '@tensorflow/tfjs';
// 可视化库：@tensorflow/tfjs-vis
// 用parcel打包工具运行：parcel + 文件夹名
window.onload = async () => {
    const xs = [1,2,3,4];
    const ys = [1,3,5,7];

    // 散点图
    tfvis.render.scatterplot(
        {name: '线性回归训练集'},
        { values: xs.map((x,i) => ({x, y: ys[i]})) },
        { xAxisDomain: [0,5], yAxisDomain: [0,8]  }
    );

    // 定义单层单个神经元组成的神经网络：
    // 1。初始化一个神经网络模型
    // 2。 为神经网络模型添加层
    // 3。 设计层的神经元个数和inputShape
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // 损失函数： 均方误差MSE
    // 优化器： 随机梯度下降SGD
    model.compile({loss: tf.losses.meanSquaredError, optimizer:tf.train.sgd(0.1)});
    
    // 训练模型及可视化训练过程：
    // 1 将训练数据转为Tensor
    // 2 训练模型
    // 3 使用tfvis可视化训练过程
    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);
    await model.fit(inputs, labels,{
        batchSize: 4,
        epochs: 100,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    });

    // 进行预测：
    // 1. 将待预测数据转为Tensor
    // 2. 使用训练好的模型进行预测
    // 3. 将输出的Tensor转为普遍数据并显示
    const output = model.predict(tf.tensor([5])); // 预测输入5，结果tensor是多少
    // output.print();
    // output.dataSync[0];//tensor转化为数字
    alert(`如果x = 5, 那么预测的y = ${output.dataSync()[0]}`)
};

