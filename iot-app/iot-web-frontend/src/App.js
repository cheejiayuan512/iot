import React, {useEffect, useRef, useState} from "react";
import "./styles.scss";
import { FileUploader } from "react-drag-drop-files";

const fileTypes = ["JPG", "JPEG"];
const App = () => {
    const videoRef = useRef(null);
    const photoRef = useRef(null);
    const stripRef = useRef(null);
    const colorRef = useRef(null);
    const [expression, setExpression] = useState('Please take a photo or upload an image!')
    const [file, setFile] = useState(null);
    const handleChange = (file) => {
        setFile(file);
        console.log(file)
        let strip = stripRef.current;
        const reader = new FileReader();
        reader.onloadend = () => {
            console.log(reader.result);
            // Logs data:<type>;base64,wL2dvYWwgbW9yZ...
            const file_data = reader.result;
            fetch('https://iot-fyp.loophole.site/predict_image', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({"user_photo": file_data.split(',')[1]})
            })
                .then(response => response.text().then(r => {
                    console.log(r)
                    setExpression(r)
                    const link = document.createElement("div");
                    link.href = file_data;
                    link.innerHTML = `<div class="m-2"><img src='${file_data}' alt='thumbnail'/><div class="m-2">This is a ${r}</div></div>`;
                    strip.insertBefore(link, strip.firstChild);
                }))


        };
        reader.readAsDataURL(file);
    };
    useEffect(() => {
        getVideo();
    }, [videoRef]);

    const getVideo = () => {
        navigator.mediaDevices
            .getUserMedia({video: {width: 480, facingMode:'environment'}})
            .then(stream => {
                let video = videoRef.current;
                video.srcObject = stream;
                video.play();
            })
            .catch(err => {
                console.error("error:", err);
            });
    };

    const paintToCanvas = () => {
        let video = videoRef.current;
        let photo = photoRef.current;
        let ctx = photo.getContext("2d");

        const width = 320;
        const height = 240;
        photo.width = width;
        photo.height = height;

        return setInterval(() => {
            let color = colorRef.current;

            ctx.drawImage(video, 0, 0, width, height);
            let pixels = ctx.getImageData(0, 0, width, height);

            // color.style.backgroundColor = `rgb(${pixels.data[0]},${pixels.data[1]},${
            //     pixels.data[2]
            // })`;
            // color.style.borderColor = `rgb(${pixels.data[0]},${pixels.data[1]},${
            //     pixels.data[2]
            // })`;
        }, 200);
    };

    const takePhoto = () => {
        let photo = photoRef.current;
        let strip = stripRef.current;

        const data = photo.toDataURL("image/jpeg");

        console.warn(data);
        fetch('https://iot-fyp.loophole.site/predict_image', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"user_photo": data.split(',')[1]})
        })
            .then(response => response.text().then(r => {
                console.log(r)
                setExpression(r)
                const link = document.createElement("div");
                link.href = data;
                // link.setAttribute("download", "myWebcam");
                link.innerHTML = `<div class="m-2"><img src='${data}' alt='thumbnail'/><div class="m-2">This is a ${r}</div></div>`;
                strip.insertBefore(link, strip.firstChild);
            }))

    };

    return (
        <div className="container">
            <h1>Cat App</h1>
            <div className="webcam-video">
                <video
                    onCanPlay={() => paintToCanvas()}
                    ref={videoRef}
                    className="player"
                />
                <canvas ref={photoRef} className="photo" hidden/>
                <div className="photo-booth">
                    <div ref={stripRef} className="strip"/>
                </div>
            </div>
            <button type="button" className="btn btn-primary m-2" onClick={() => takePhoto()}>Take a photo</button>

            <FileUploader handleChange={handleChange} name="file" types={fileTypes} />
            <a>{expression}</a>
        </div>
    );
};

export default App;
