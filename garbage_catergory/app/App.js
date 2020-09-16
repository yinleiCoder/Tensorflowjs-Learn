import React, {PureComponent} from 'react'
import {Button} from 'antd';
import 'antd/dist/antd.css'
import * as tf from '@tensorflow/tfjs'

const DATA_URL = 'http://127.0.0.1:8080/'

class App extends PureComponent{

    state = {}

    async componentDidMount() {
        this.model = await tf.loadLayersModel(DATA_URL+'/model.json');//yarn add global http-server,then "hs 模型文件夹 --cors"
        this.model.summary();
        this.CLASSES = await fetch(DATA_URL + '/classes.json').then(res=>res.json())
    }

    // 将文件变为图片
    file2img(file) {
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload=(e)=>{
                const img = document.createElement('img')
                img.src = e.target.result // base64
                img.width = 224
                img.height = 224
                img.onload = () => {
                    resolve(img)
                }
            }
        })
    }

    // 将图形转为tensor
    img2x(imgEl) {
        return tf.tidy(()=>{
            return tf.browser.fromPixels(imgEl)
                    .toFloat().sub(255 / 2).div(255 / 2)
                    .reshape([1, 224,224,3]);
        })
    }

    predict = async (file) => {
        const img = await this.file2img(file);
        this.setState({imgSrc: img.src})
        setTimeout(() => {
            const pred = tf.tidy(()=>{
                const x = this.img2x(img)
                return this.model.predict(x);
            })
            pred.print();
            const result = pred.arraySync()[0].map((score, i) => ({score, label: this.CLASSES[i]})).sort((a,b)=>b.score-a.score)
            console.log(result)
            this.setState({result})
        }, 0);
    }

    render() {
        const {imgSrc} = this.state;
        return(
            <div style={{padding: 20}}>
                <Button type="primary"
                    size="large"
                    style={{width:'100%'}}
                    onClick={()=>this.upload.click()}
                >垃圾分类</Button>
                <input type="file"
                    onChange={e =>this.predict(e.target.files[0])}
                    ref={el => {this.upload = el;}}
                    style={{display: 'none'}}
                />
                {imgSrc && <div style={{marginTop: 20, textAlign: 'center'}}>
                    <img src={imgSrc} style={{maxWidth: '100%', maxHeight: '300px'}}/>
                    </div>}
            </div>
        );
    }
}

export default App;