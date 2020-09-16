const fs = require('fs')
const tf = require('@tensorflow/tfjs-node');


const img2x = (imgPath) => {
    const buffer = fs.readFileSync(imgPath)
    return tf.tidy(()=>{
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
        const imgTsResized = tf.image.resizeBilinear(imgTs, [224, 224]);
        // 归一化
        return imgTsResized.toFloat().sub(255/2).div(255/2).reshape([1,224,224,3]);
    });
}

// const img2x = (buffer) => {
//     return tf.tidy(()=>{
//         const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
//         const imgTsResized = tf.image.resizeBilinear(imgTs, [224, 224]);
//         // 归一化
//         return imgTsResized.toFloat().sub(255/2).div(255/2).reshape([1,224,224,3]);
//     });
// }

// 读取数据
const getData = async (trainDir, outputDir) => {
    // 读取垃圾类别数据:厨余垃圾、可回收垃圾、其他垃圾、有害垃圾
    const classes = fs.readdirSync(trainDir).filter(n=>!n.includes('.')); 
    fs.writeFileSync(`${outputDir}/classes.json`,JSON.stringify(classes) );
    // const inputs = [];
    // const labels = [];
    const data = [];
    classes.forEach((dir, dirIndex) => {
        fs.readdirSync(`${trainDir}/${dir}`).filter(n => n.match(/jpeg$/)).forEach(filename => {
            console.log(dir,filename)//读取目录下的filename
            const imgPath = `${trainDir}/${dir}/${filename}`;
            data.push({imgPath, dirIndex})
            // const buffer = fs.readFileSync(imgPath) // 读取到内存中
            // const x = img2x(buffer);
            // inputs.push(x);
            // labels.push(dirIndex);
        });
    });

    tf.util.shuffle(data);// 打乱数据

    const ds = tf.data.generator(function* (){// 生成器函数
        // 分批读取
        const count = data.length;
        const batchSize = 32;
        for(let start=0;start < count; start+=batchSize){
            const end = Math.min(start+batchSize, count);
            yield tf.tidy(()=>{
                const inputs = [];
                const labels = [];
                for(let j=start;j<end;j+=1){
                    const {imgPath, dirIndex} = data[j];
                    const x = img2x(imgPath);
                    inputs.push(x);
                    labels.push(dirIndex);
                }
                const xs = tf.concat(inputs);
                const ys = tf.tensor(labels);
                return {xs, ys}
            });
        }
    });
    return {
        ds,
        classes
    }
    // const xs = tf.concat(inputs);
    // const ys = tf.tensor(labels);
    // return {
    //     xs,
    //     ys,
    //     classes
    // };
};

module.exports =  getData;