import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
    
    // 归一化(压缩)任务： 把大数量级特征转化到较小的数量级下，通常是[0,1] 或[-1,1]
    // e.g: 身高体重预测、房价预测

    // 归一化的原因： 
    // 1. 绝大多数tensorflow.js的模型都不是给特别大的数设计的。
    // 2。 将不同数量级的特征转换到同一数量级，防止某个特征影响过大。
    const heights = [150,160,170];
    const weights = [40,50,60];

    // 散点图
    tfvis.render.scatterplot(
        {name: '身高体重训练数据'},
        {values: heights.map((x,i) => ({x,y: weights[i]}) ) },
        {
            xAxisDomain: [140,180],
            yAxisDomain: [30, 70]
        }
    );

    // 归一化压缩: [0,1]
    const inputs = tf.tensor(heights).sub(150).div(20);//数据-150 / 20 => 0~1
    // inputs.print();
    const labels = tf.tensor(weights).sub(40).div(20);
    // labels.print();
    
    // 训练数据模型
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1)});

    await model.fit(inputs, labels,{
        batchSize: 3,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    });
    const output = model.predict(tf.tensor([180]).sub(150).div(20)); // 180cm高的人
    // 反归一化： [0,1] => 变为原值
    alert(`如果身高为 180cm，那么预测体重为 ${output.mul(20).add(40).dataSync()[0]}kg`);

};