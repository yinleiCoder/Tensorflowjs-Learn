const getData = require('./data');
const tf = require('@tensorflow/tfjs-node')

const  TRAIN_DIR = '../垃圾分类/train'
const OUTPUT_DIR = '../garbage_output'

const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
const main = async () => {
    // 加载数据
    const {
        // xs,
        // ys,
        ds,
        classes
    } = await getData(TRAIN_DIR,OUTPUT_DIR);
    // console.log(xs, ys, classes);


    // 定义模型
    // mobilenet 模型
    const mobilenet = await tf.loadLayersModel(MOBILENET_URL);
    mobilenet.summary();
    // 截断处输出可知是86层
    // const layer = mobilenet.layers.map((l, i) => [l.name, i])
    const model = tf.sequential();
    for(let i =0; i<=86;i+=1){
        const layer = mobilenet.layers[i];
        layer.trainable = false;
        model.add(layer);
    }
    //双层神经网络
    model.add(tf.layers.flatten())
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu',
    }));
    model.add(tf.layers.dense({
        units: classes.length,
        activation: 'softmax'
    }));

    // 训练模型
    model.compile({
        loss: 'sparseCategoricalCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['acc']
    });
    // await model.fit(xs, ys, {
    //     epochs: 20,
    // });
    await model.fitDataset(ds, {
        epochs: 20,
    });
    await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`);
}

main();