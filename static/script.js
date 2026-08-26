function sendMessage(){

    let input=document.getElementById("user-input");

    let message=input.value;

    if(message=="") return;

    let chat=document.getElementById("chat-box");

    chat.innerHTML+="<div class='user'><b>You:</b> "+message+"</div>";

    fetch("/get",{

        method:"POST",

        headers:{
            "Content-Type":"application/x-www-form-urlencoded"
        },

        body:"msg="+encodeURIComponent(message)

    })

    .then(response=>response.json())

    .then(data=>{

        chat.innerHTML+="<div class='bot'><b>Bot:</b> "+data.reply+"</div>";

        chat.scrollTop=chat.scrollHeight;

    });

    input.value="";
}