import * as tfvis from '@tensorflow/tfjs-vis'
// 基于迁移学习的语音识别器：声控轮播图
// 训练中文的语音识别器

// 需要静态服务器
import * as speechCommands from '@tensorflow-models/speech-commands'
const MODEL_PATH = "http://127.0.0.1:8080";
let transferRecognizer;
window.onload = async () => {
// 步骤：
// 1. 从浏览器中收集中文语音训练数据
    // 使用speech commands创建迁移学习器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'
    );
    await recognizer.ensureModelLoaded();
    transferRecognizer = recognizer.createTransfer('轮播图'); //创建迁移学习器
    // 编写前端界面准备收集语音数据
    // 调用collectExample方法收集语音数据

// 2. 使用speech commands包进行迁移学习
        // 语音识别迁移学习的训练和预测
        // 编写界面调用train()来进行迁移学习训练
        // 编写界面调用listen()从浏览器监听麦克风预测
        // 编写界面调用stopListen()停止监听
// 3. 语音训练数据的保存和加载（非模型的保存）
        // 保存语音训练数据到文件
        // 加载语音训练数据到模型
        // 通过加载的语音数据进行训练和预测


}

window.collect = async (btn) => {
    btn.disabled = true;
    const label = btn.innerText;
    // console.log(label)
    // 调用collectExample方法收集语音数据
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );
    btn.disabled = false;
    console.log(transferRecognizer.countExamples());
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(),null, 2);
}

//使用speech commands包进行迁移学习
window.train = async () =>{
    await transferRecognizer.train({
        epochs: 200,
        callback: tfvis.show.fitCallbacks(
            {name: '训练效果'},
            ['loss', 'acc'],
            {callbacks: ['onEpochEnd']}
        )
    });
}

window.toggle = async (checked) => {
    // console.log(checked)
    if(checked) {
        await transferRecognizer.listen(result => {
            const {scores} = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            console.log(labels[index])
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.75
        });
    } else {
        transferRecognizer.stopListening();
    }
}

// 保存数据并序列化后下载二进制文件(模拟a标签下载),把下载好后的二进制文件放到静态服务器下的slider文件夹下
window.save = () => {
    const arrayBuffer = transferRecognizer.serializeExamples();
    // arrayBuffer下载成二进制文件
    const blob = new Blob([arrayBuffer]);
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'data.bin';
    link.click();
}
