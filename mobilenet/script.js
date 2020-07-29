import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import {file2img} from './utils'
import {IMAGENET_CLASSES} from './imagenet_classes'
// 使用预训练模型进行图片的分类

// mobilenet是一种卷积神经网络，轻量
// 也需要开启静态服务器:http-server data --cors 如果cors不起作用，可以ctrl+f5强制刷新浏览器缓存
// 启动整个程序使用打包工具：parcel 文件夹/*ml
// 使用tensorflow.js的loadLayersModel方法加载模型
const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
window.onload = async () => {
    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    // console.log(model);
    window.predict = async (file) => { //需要file转为htmlelement
        const img = await file2img(file);
        document.body.appendChild(img);
        // 转化为tensor
        const pred = tf.tidy(()=>{//tidy是为了清除中间的tensor,优化webgl内存
            const input = tf.browser.fromPixels(img)
            .toFloat()
            .sub(255/2)
            .div(255/2) // 归一化
            .reshape([1, 224, 224, 3]);
            return model.predict(input);
        });
        // pred.print();
        const index = pred.argMax(1).dataSync()[0];
        // console.log(index);
        setTimeout(() => {
            alert(`预测结果：${IMAGENET_CLASSES[index]}`) 
        }, 0);
    };
};