const mongoose = require('mongoose')
const {Schema} = mongoose


const studentSchema = new Schema({
    name: String,
    email:{
        type: String,
        unique: false
    },
    rollno:{
        type: String,
        unique: true
    }
})

const studentModel = mongoose.model('Student', studentSchema)

module.exports = studentModel
