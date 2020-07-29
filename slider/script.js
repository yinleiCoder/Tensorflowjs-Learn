import * as speechCommands from '@tensorflow-models/speech-commands'
const MODEL_PATH = "http://127.0.0.1:8080";
let transferRecognizer;
let curIndex = 0;
window.onload = async () => {
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'
    );
    await recognizer.ensureModelLoaded();
    transferRecognizer = recognizer.createTransfer('轮播图'); //创建迁移学习器
    // 加载从speech-cn下保存的采集的语音数据
    const res =  await fetch(MODEL_PATH + '/slider/data.bin');
    const arrayBuffer = await res.arrayBuffer(); // 拿到arrayBuffer
    // console.log(arrayBuffer)
    transferRecognizer.loadExamples(arrayBuffer);
    console.log(transferRecognizer.countExamples());
    await transferRecognizer.train({
        epochs: 100
    });
    console.log('done');
}

window.toggle = async (checked) => {
    // console.log(checked)
    if(checked){
        await transferRecognizer.listen(result => {
            const {scores} = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            console.log(labels[index])
            window.play(labels[index]);
        },{
            overlapFactor: 0,
            probabilityThreshold: 0.55
        });
    }else {
        transferRecognizer.stopListening();
    }
}

window.play = (label) => {
    const div = document.querySelector('.slider > div');
    if(label==='上一张') {
        if(curIndex === 0){
            return;
        }
        curIndex -= 1;
    } else {
        if(curIndex === document.querySelectorAll('img').length - 1){
            return;
        }
        curIndex += 1;
    }
    console.log(curIndex)
    div.style.transition = 'transform 1s';
    div.style.transform = `translateX(-${100*curIndex}%)`
}