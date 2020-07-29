import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {MnistData} from './data'
window.onload = async () =>{
    // 卷积神经网络
// 加载MNIST数据集: 首先需要起一个静态服务器： hs data --cors
// parcel mnist/*ml 不要打包data文件夹了
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);
    // console.log(examples)
    // 将tensor转化为图片展示到浏览器上
    const surface = tfvis.visor().surface({name: '输入示例'});
    for(let i=0;i<20;i+=1) {
        const imageTensor =tf.tidy(()=>{// tidy防止数据量过大，有助于清除内存防止内存泄漏
            return examples.xs.slice([i,0],[1,784]).reshape([28, 28, 1]);
        }); 
        //tensor转化为像素
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style='margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);
        // document.body.appendChild(canvas);
        surface.drawArea.appendChild(canvas);
    }
// 构建卷积神经网络并训练
    const model = tf.sequential();
    // 1.卷积层
    model.add(tf.layers.conv2d({
        inputShape: [28,28,1],
        kernelSize: 3,//卷积核大小
        filters: 8,
        strides: 1,//移动步长
        activation: 'relu', //relu激活函数可以移除掉不常用的特征
        kernelInitializer:'varianceScaling' // 卷积核初始方法，可以加快收敛速度
    }));// 这里的图片是2维的
    // 2. 池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2,2]
    }));
    // 组合上面的特征
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu', 
        kernelInitializer:'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2,2]
    }));
    // 3.全连接层
    model.add(tf.layers.flatten());//dense层是1维，将高维数据展平
    model.add(tf.layers.dense({
        units: 10, // 手写数字图片是0-9个数字
        activation: 'softmax', // 多分类的激活函数
        kernelInitializer:'varianceScaling'
    }));

// 训练模型、准备训练集和验证集
    model.compile({
        loss: 'categoricalCrossentropy', //交叉熵
        optimizer: tf.train.adam(),
        metrics: 'accuracy'
    });
    // 训练集和验证集
    const [trainXs, trainYs] = tf.tidy(()=>{
        const d = data.nextTrainBatch(1000);
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ]
    });
    const [testXs, testYs] = tf.tidy(()=>{
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels,
        ]
    });
    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练效果'},
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {callbacks: ['onEpochEnd']} //只显示onEpochEnd图表
        )
    });
// 使用Canvas绘制数字并预测: 编写前端界面输入待预测数据-》使用训练好的模型进行预测-》 将输出的tensor转为普通数据并显示
    const canvas = document.querySelector('canvas');
    canvas.addEventListener('mousemove', (e)=>{
        if(e.buttons === 1){// 鼠标左键
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX,e.offsetY, 22,22);
        }
    });
    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0,0, 300,300);
    }
    clear();
    window.predict = () => {
        const input = tf.tidy(()=>{
            return tf.image.resizeBilinear(
                tf.browser.fromPixels(canvas),
                [28, 28],
                true
            ).slice([0,0,0], [28,28,1])
            .toFloat()
            .div(255)
            .reshape([1, 28, 28, 1]);//归一化并将canvas转化为tensor
        });
        const pred = model.predict(input).argMax(1);
        alert(`预测结果：${pred.dataSync()[0]}`)        
    }

};



