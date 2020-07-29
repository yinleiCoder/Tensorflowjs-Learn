import {getInputs} from './data'
import * as tfivs from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import {img2x, file2img} from './utils'
const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];
//迁移学习的图片分类器
window.onload = async ()=>{
// - 加载商标训练数据并可视化
    const {inputs, labels} = await getInputs();
    const surface = tfivs.visor().surface({
        name: '输入示例',
        styles: {height: 350}
    });
    // console.log(inputs,labels);
    inputs.forEach(imgEl => {
        surface.drawArea.appendChild(imgEl);
        // document.body.appendChild(imgEl);
    });
// - 定义模型结构：截断模型+双层神经网络
    // 加载预训练模型MobileNet模型并截断
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    mobilenet.summary();
    const layer = mobilenet.getLayer('conv_pw_13_relu')
    const truncatedMobilenet = tf.model({//截断
        inputs: mobilenet.inputs,
        outputs: layer.output
    });
    // 构建双层神经网络
    const model = tf.sequential();
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)
    }));
    model.add(
        tf.layers.dense({
            units: 10,
            activation: 'relu'
        })
    );
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));
    // 截断模型作为输入，双层神经网络作为输出

// - 迁移学习下的模型训练
    // 定义损失函数和优化器
    model.compile({
        loss: 'categoricalCrossentropy', //交叉熵
        optimizer: tf.train.adam()
    });
    // 训练数据经过截断模型，转为可以用于新模型训练的数据
    const {xs, ys} = tf.tidy(()=>{
        const xs = tf.concat(inputs.map(imgEl =>truncatedMobilenet.predict(img2x(imgEl))));
        const ys = tf.tensor(labels);
        return {xs, ys};
    });
    // 使用tensorflowjs的fit()进行训练
    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfivs.show.fitCallbacks(
            {name: '训练效果'},
            ['loss'],
            {callbacks: ['onEpochEnd']}
        )
    })
// - 迁移学习下的模型预测
    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const x = img2x(img);
            const input = truncatedMobilenet.predict(x);
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${BRAND_CLASSES[index]}`);
        }, 0);
    };

    window.download = async () => {
        await model.save('downloads://model');
    };
}