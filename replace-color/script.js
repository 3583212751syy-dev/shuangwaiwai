Coloris({
        el: '.coloris',
        onChange: (color) => changeNewColor(color),
        format: 'hex'
});

var inputImg = document.getElementById('uploadInput');
var canvas = document.getElementById('Canvas');
var context = canvas.getContext('2d');
var newColor = '000000';
var tolerance = 100;
var oldImg;

function changeNewColor(c){
        newColor = c.substring(1, c.length);
}

inputImg.onchange = function(e) {
        var img = new Image();
        img.onload = drawImg;
        img.onerror = failed;
        img.src = URL.createObjectURL(this.files[0]);
        oldImg = img;
};

function drawImg(){
        canvas.width = this.width;
        canvas.height = this.height;

        context.drawImage(this, 0, 0);
}

function failed(){
        alert("Please input an image file.");
}

canvas.addEventListener('click', function(e){
        var eventLocation = getEventLocation(this,e);
        changePixels(eventLocation.x, eventLocation.y, newColor, tolerance);
},false);

function getPixel(imageData, x, y) {
        if (x < 0 || y < 0 || x >= imageData.width || y >= imageData.height) {
          return '000000';  // impossible color
        } else {
          const offset = (y * imageData.width + x) * 4;
          let rgb = imageData.data.slice(offset, offset + 3);
          return rgbToHex(rgb[0], rgb[1], rgb[2]);
        }
}

function colorsMatch(a, b, rangeSq) {
        let rgbA = hexToRgb(a);
        let rgbB = hexToRgb(b);
        const dr = rgbA[0] - rgbB[0];
        const dg = rgbA[1] - rgbB[1];
        const db = rgbA[2] - rgbB[2];
        //const da = a[3] - b[3];
        return dr * dr + dg * dg + db * db < rangeSq;
}

function changePixels(x, y, color, tol){
        const imgData = context.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        let oldColor = getPixel(imgData, x, y);
        
        for(let row = 0; row < canvas.height; row++){
                for(let col = 0; col < canvas.width; col++){
                        let index = (col + (row * imgData.width))*4;
                        let r = imgData.data[index];
                        let g = imgData.data[index + 1];
                        let b = imgData.data[index + 2];
                        let a = imgData.data[index + 3];
                        let hex = rgbToHex(r, g, b, a);
                        if(hex != '-1'){
                                if(colorsMatch(hex, oldColor, tol*tol)){
                                        let newRgb = hexToRgb(color);
                                        imgData.data[index] = newRgb[0];
                                        imgData.data[index + 1] = newRgb[1];
                                        imgData.data[index + 2] = newRgb[2];
                                }
                        }
                }
        }
        
        context.putImageData(imgData, 0, 0);
}

function colorToHex(color){
        var hex = color.toString(16);
        return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b, a=1) {
        if (r > 255 || g > 255 || b > 255)
            alert("invalid color component");
        if(a == 0){
                return '-1';
        }
        return colorToHex(r) + colorToHex(g) + colorToHex(b);
}

function hexToRgb(hex){
        if(hex.length != 6){
                alert(hex + ": 6digit");
                //alert("Only six-digit hex colors are allowed.");
        }
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        
        // return {r, g, b} 
        return [r, g, b ];
        
}

function getElementPosition(obj) {
        var curleft = 0, curtop = 0;
        if (obj.offsetParent) {
            do {
                curleft += obj.offsetLeft;
                curtop += obj.offsetTop;
            } while (obj = obj.offsetParent);
            return { x: curleft, y: curtop };
        }
        return undefined;
}
function getEventLocation(element,event){
        // Relies on the getElementPosition function.
        var pos = getElementPosition(element);
        
        return {
                x: (event.pageX - pos.x),
                  y: (event.pageY - pos.y)
        };
}

document.getElementById('download').addEventListener('click', function(e){
        let canvasUrl = canvas.toDataURL();
        const createEl = document.createElement('a');
        createEl.href = canvasUrl;
        createEl.download = "color-replaced";
        createEl.click();
        createEl.remove();
});

document.getElementById("tolerance").onchange = function(e) {
        tolerance = document.getElementById("tolerance").value;
};

document.getElementById('revert').addEventListener('click', function(e){
        context.drawImage(oldImg, 0, 0, canvas.width, canvas.height);
});