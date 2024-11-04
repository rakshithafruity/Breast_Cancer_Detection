import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import styles from '../styles/home.module.css';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Switch from '@mui/material/Switch';
import { styled } from '@mui/material/styles';
import Typography from '@mui/material/Typography';

const MaterialUISwitch = styled(Switch)(({ theme }) => ({
  width: 80,
  height: 40,
  padding: 7,
  '& .MuiSwitch-switchBase': {
    margin: 1,
    padding: 0,
    transform: 'translateX(6px)',
    '&.Mui-checked': {
      color: '#fff',
      transform: 'translateX(38px)',
      '& .MuiSwitch-thumb:before': {
        backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="${encodeURIComponent(
          '#FFFFFF',
        )}"><path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm40-80h480L570-480 450-320l-90-120-120 160Zm-40 80v-560 560Z"/></svg>')`,
      },
      '& + .MuiSwitch-track': {
        opacity: 1,
        backgroundColor: theme.palette.mode === 'dark' ? '#8796A5' : '#aab4be',
      },
    },
  },
  '& .MuiSwitch-thumb': {
    backgroundColor: theme.palette.mode === 'dark' ? '#003892' : '#001e3c',
    width: 37,
    height: 35,
    '&::before': {
      content: "''",
      position: 'absolute',
      width: '100%',
      height: '100%',
      left: 0,
      top: 0,
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'center',
      backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="${encodeURIComponent(
        '#FFFFFF',
      )}"><path d="m380-300 280-180-280-180v360ZM160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h640q33 0 56.5 23.5T880-720v480q0 33-23.5 56.5T800-160H160Zm0-80h640v-480H160v480Zm0 0v-480 480Z"/></svg>')`,
    },
  },
  '& .MuiSwitch-track': {
    opacity: 1,
    backgroundColor: theme.palette.mode === 'dark' ? '#8796A5' : '#aab4be',
    borderRadius: 50 / 2,
  },
}));



export default function Home() {

  const [checked, setChecked] = useState(false);
  const [input, setInput] = useState('video');

  const[classLLMfile, setClassLLMfile] = useState('');
  const[classDLfile, setClassDLfile] = useState('');
  const[segLLMfile, setSegLLMfile] = useState('');
  const[segDLfile, setSegDLfile] = useState('');
  const[maskLLMfile, setMaskLLMfile] = useState('');
  const[maskDLfile, setMaskDLfile] = useState('');
  const navigate = useNavigate();
  const UnrealUrl ="http://localhost:5050"
  const PixelUrl="http://localhost:80"

  const handleFileChange = (e, setFile) => {
    setFile('')
    const file = e.target.files[0]
    if(file){
      const fileUrl = URL.createObjectURL(file);
      setFile(fileUrl);
    }
  };

  const uploadFileClass = async (fileUrl, predictUrl, flag) => {
    const toastID = toast.loading(`Uploading ${input} and Predicting...`, { duration: Infinity });
    try {
      const formData = new FormData();
      const response = await fetch(fileUrl);
      const blob = await response.blob();
      if(input==='video'){
        formData.append('video', blob, 'video.mp4');
      }
      else{
        formData.append('image', blob, 'image.png');
      }
      const predictResponse = await axios.post(predictUrl, formData, {
        withCredentials: true,
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        responseType: `${checked ? 'arraybuffer' : 'blob'}`,
      });

      const predictBlob = new Blob([predictResponse.data], { type: checked ? 'image/png' : 'video/mp4' });
      toast.dismiss(toastID);
      toast.success('Prediction complete');

      const UnrealFormData = new FormData();
      UnrealFormData.append(`${input}_${flag}_classification`, predictBlob, `${input}_${flag}_classification.${checked ? 'png' : 'mp4'}`);
      UnrealFormData.append(`input_${input}_${flag}`, blob, `input_${input}_${flag}.${checked ? 'png' : 'mp4'}`);

      const postUrl = `${UnrealUrl}/${checked ? 'upload_image' : 'upload_video'}`;
      toast.loading(`Uploading ${input} to Unreal...`, { duration:'2s' });
      await fetch(postUrl, {
        method: 'POST',
        body: UnrealFormData,
        headers: {
          'Accept': 'application/json',
        },
      });
      window.open(PixelUrl, '_blank')
    } catch (error) {
      toast.dismiss(toastID);
      if (error.response && error.response.status === 401) {
        toast.error('Unauthorized access. Please login.');
        navigate('/login');
      } else {
        toast.dismiss(toastID)
        if(!fileUrl) toast.error(`Please upload ${input}`)
        else toast.error(`Failed to upload ${input}`);
      }
    }
  };

  const uploadFileSeg = async(fileUrl, maskUrl, predictUrl, flag) => {
    console.log('hi')
    const toastID = toast.loading(`Uploading ${input} and Predicting...`, { duration: Infinity });
    try {
      const formData = new FormData();
      const res1 = await fetch(fileUrl);
      const res2 = await fetch(maskUrl);
      const blob1 = await res1.blob();
      const blob2 = await res2.blob()
      formData.append('image', blob1, 'image.jpeg')
      formData.append('mask', blob2, 'mask.jpeg')
      const predictResponse = await axios.post(predictUrl, formData, {
        withCredentials: true,
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        responseType: `arraybuffer`,
      });
      const predictBlob = new Blob([predictResponse.data], { type: checked ? 'image/png' : 'video/mp4' });
      toast.dismiss(toastID);
      toast.success('Prediction complete');

      const UnrealFormData = new FormData();
      UnrealFormData.append(`${input}_${flag}_segmentation`, predictBlob, `${input}_${flag}_segmentation.${checked ? 'png' : 'mp4'}`);
      UnrealFormData.append(`input_${input}_${flag}`, blob1, `input_${input}_${flag}.${checked ? 'png' : 'mp4'}`);

      const postUrl = `${UnrealUrl}/${checked ? 'upload_image' : 'upload_video'}`;
      toast.loading(`Uploading ${input} to Unreal...`, { duration:'2s' });
      await fetch(postUrl, {
        method: 'POST',
        body: UnrealFormData,
        headers: {
          'Accept': 'application/json',
        },
      });

      window.open(PixelUrl, '_blank')
    } catch (error) {
      toast.dismiss(toastID);
      if (error.response && error.response.status === 401) {
        toast.error('Unauthorized access. Please login.');
        navigate('/login');
      } else {
        toast.dismiss(toastID)
        console.log(error)
        if(!fileUrl) toast.error(`Please upload ${input}`)
        else toast.error(`Failed to upload ${input}`);
      }
    }
  };

  const handleChange = (event) => {
    setClassLLMfile('');
    setClassDLfile('');
    setSegLLMfile('');
    setSegDLfile('');
    setMaskLLMfile('');
    setMaskDLfile('');
    setChecked(event.target.checked);
    setInput(event.target.checked ? 'image' : 'video');
  };





  return (
    <div className="flex flex-col items-center justify-center mt-28 h-full">
      <h1 className="text-3xl font-bold mb-8">Breast Cancer Segmentation and Classification with LLM and DL</h1>
      <div className="w-full max-w-5xl flex flex-col lg:flex-row justify-center items-center mb-8">
      <h2 className="text-2xl font db-8 items-left mr-2">Select your type of input: </h2>
      <Typography variant="h5" style={{ fontWeight: 'bold' }}>{checked ? 'Image' : 'Video'}</Typography>
      <MaterialUISwitch checked={checked} onChange={handleChange} />
      </div>
      <div className="flex flex-wrap w-full max-w-5xl mx-auto">
        <div className="w-full md:w-1/2 px-2">
          <div className="bg-white rounded-xl overflow-hidden shadow-lg h-full">
            <div className="px-6 py-4">
              <div className="font-bold text-xl mb-2">Classification with LLM</div>
              <div className="flex flex-col mt-4">
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setClassLLMfile);}} style={{ marginBottom: '10px' }} />
                {classLLMfile && (
                  <div>
                    {checked ? <img src={classLLMfile} alt="Uploaded Image" /> : <video src={classLLMfile} controls/>}
                  </div>
                )}
                <div className="flex mt-2">
                  <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-2" onClick={() => uploadFileClass(classLLMfile, checked ? '/image_class_dl' : '/video_class_llm', 'llm')}>Predict with LLM</button>
                  <button className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" onClick={() => {setClassLLMfile('');}}>Clear</button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="w-full md:w-1/2 px-2">
          <div className="bg-white rounded-xl overflow-hidden shadow-lg h-full">
            <div className="px-6 py-4">
              <div className="font-bold text-xl mb-2">Classification with DL</div>
              <div className="flex flex-col mt-4">
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setClassDLfile);}} style={{ marginBottom: '10px' }} />
                {classDLfile && (
                  <div>
                    {checked ? <img src={classDLfile} alt="Uploaded Image" /> : <video src={classDLfile} controls/>}
                  </div>
                )}
                <div className="flex mt-2">
                  <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-2" onClick={() => uploadFileClass(classDLfile, checked ? '/image_class_dl' : '/video_class_dl', 'dl')}>Predict with DL</button>
                  <button className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" onClick={() => {setClassDLfile('');}}>Clear</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* SEGMENTATION */}
        <div className="w-full md:w-1/2 px-2 mt-4">
          <div className="bg-white rounded-xl overflow-hidden shadow-lg h-full">
            <div className="px-6 py-4">
              <div className="font-bold text-xl mb-2">Segmentation with LLM</div>
              <div className="flex flex-col mt-4">
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setSegLLMfile)}} style={{ marginBottom: '10px' }} />
                {segLLMfile && (
                  <div>
                    {checked ? <img src={segLLMfile} alt="Uploaded Image" /> : <video src={segLLMfile} controls/>}

                  </div>
                )}
                <h3 className="font-bold text-lg mb-2 flex justify-left mt-4">Ground Truth Input: </h3>
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setMaskLLMfile)}} style={{ marginBottom: '10px' }} />
                {maskLLMfile && (
                  <div>
                    {checked ? <img src={maskLLMfile} alt="Uploaded Image" /> : <video src={maskLLMfile} controls/>}
                  </div>
                )}
                <div className="flex mt-2">
                  <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-2" onClick={()=> uploadFileSeg(segLLMfile, maskLLMfile, '/image_seg_llm', 'llm')}>Predict with LLM</button>
                  <button className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" onClick={() => {setSegLLMfile(''); setMaskLLMfile('');}}>Clear</button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="w-full md:w-1/2 px-2 mt-4">
          <div className="bg-white rounded-xl overflow-hidden shadow-lg h-full">
            <div className="px-6 py-4">
              <div className="font-bold text-xl mb-2">Segmentation with DL</div>
              <div className="flex flex-col mt-4">
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setSegDLfile)}} style={{ marginBottom: '10px' }} />
                {segDLfile && (
                  <div>
                    {checked ? <img src={segDLfile} alt="Uploaded Image" /> : <video src={segDLfile} controls/>}
                  </div>
                )}
                <h3 className="font-bold text-lg mb-2 flex justify-left mt-4">Ground Truth Input: </h3>
                <input type="file" accept={checked ? 'image/*' : 'video/*'} onChange={(e) => {handleFileChange(e, setMaskDLfile)}} style={{ marginBottom: '10px' }} />
                {maskDLfile && (
                  <div>
                    {checked ? <img src={maskDLfile} alt="Uploaded Image" /> : <video src={maskDLfile} controls/>}
                  </div>
                )}
                <div className="flex mt-2">
                  <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-2" onClick={()=> uploadFileSeg(segLLMfile, maskLLMfile, '/image_seg_llm', 'llm')}>Predict with DL</button>
                  <button className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" onClick={() => {setSegDLfile(''); setMaskDLfile('');}}>Clear</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}