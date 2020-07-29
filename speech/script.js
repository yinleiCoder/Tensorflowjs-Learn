import * as speechCommands from '@tensorflow-models/speech-commands'
// 预训练模型在tfjs-models下都有。可以直接安装通过npm，而不用像以前一样手动加载通过tensorflowjs

// 这里使用的是官方库下的语音识别库
const MODEL_PATH = "http://127.0.0.1:8080/speech";
window.onload = async () => {
    // 识别器
    const recognizer = speechCommands.create(
        // 傅里叶变换
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/model.json',
        MODEL_PATH + '/metadata.json'
    );
    await recognizer.ensureModelLoaded();
    const labels = recognizer.wordLabels().slice(2); // 查看官方库预训练的模型可以识别哪些单词
    // console.log(labels)
    const resultEl = document.querySelector('#result');
    resultEl.innerHTML = labels.map(l => `
        <div>${l}</div>
    `).join('');
    // 进行语音识别
    // 1. 从浏览器监听麦克风输入
    recognizer.listen(result => {
        // console.log(result)
        const {scores} = result;
        const maxValue = Math.max(...scores);
        const index = scores.indexOf(maxValue) - 2;
        console.log(labels[index]);
        resultEl.innerHTML = labels.map((l,i) => `
            <div style="background:${i===index && '#1abc9c'};color:${i===index && 'white'};font-weight: bold;">${l}</div>
        `).join('');
    },{
        overlapFactor: 0.2,//识别频率
        probabilityThreshold: 0.9, //可能性阈值
    });
    // 2. 进行语音识别
    // 3. 编写界面显示结果
}