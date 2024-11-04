const express = require('express')
const router = express.Router()
const cors = require('cors')
// const cookieAuth = require('../middleware/cookieAuth')
const multer = require('multer')
const FormData = require('form-data')
const axios = require('axios')
const upload = multer()
const { registerUser, loginUser, verified, uploadFileClass, uploadFileSeg} = require('../controllers/authControllers')

//middleware
router.use(cors(
    {
        credentials: true,
        origin: 'http://localhost:5173'
    }
))

// router.get('/', test)
router.post('/register', registerUser)
router.post('/login', loginUser)

router.post('/video_class_llm', upload.single('video'), (req, res) => uploadFileClass(req, res, 'llm', 'video'))
router.post('/video_class_dl', upload.single('video'), (req, res) => uploadFileClass(req, res, 'dl', 'video'))

router.post('/image_class_dl', upload.single('image'), (req, res) => uploadFileClass(req, res, 'dl', 'image'))
router.post('/image_class_llm', upload.single('image'), (req, res) => uploadFileClass(req, res, 'llm', 'image'))

router.post('/image_seg_llm', upload.fields([{name: 'image', maxCount: 1}, {name:'mask', maxCount: 1}]), (req,res) => uploadFileSeg(req, res, 'llm', 'image'))

// router.post('/video_dl', cookieAuth, upload.single('video'), async (req, res) => {
//     if(res.statusCode === 401) {
//         return res.status(401).json({ error: 'Unauthorized' });
//     }
//     if (!req.file) {
//         return res.status(400).json({ message: 'No file uploaded' });
//     }
//     try {
//       const formData = new FormData();
//       formData.append('video', req.file.buffer, {
//           filename: req.file.originalname,
//           contentType: req.file.mimetype,
//       });
//       const flaskResponse = await axios.post('http://127.0.0.1:5000/predict_dl', formData, {
//           headers: {
//               ...formData.getHeaders()
//           },
//           responseType: 'stream'
//       });
//       flaskResponse.data.pipe(res);
//     } catch (error) {
//         console.log(error);
//         res.status(500).json({ message: 'Error processing image', error: error.message });
//     }
//   });

router.put('/verify/:id', verified)
// router.post('/sendotp', sendOTP)
// router.post('/verifyotp', verifyOTP)
// router.post('/upload-image', uploadImage)
module.exports = router 