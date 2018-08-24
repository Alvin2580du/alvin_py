//defind total
$(window).load(function(evt){
//$(function(){
    if (!window.console) window.console = {};
    if (!window.console.log) window.console.log = function() {};

    $('#datetimepicker').datetimepicker({
        format: 'yyyy-mm-dd',
        minView: 1,
        // maxView: 2,
        minuteStep: 5
    }).on('changeDate', function(ev){
        console.log("do change to return:" + ev.date);
        var d = toYYMMDDHH(ev.date.valueOf());
        console.log("get date info: " + d + " " + ev.date.valueOf());
        //do update form with date :
        toastr.info('数据加载中，请稍后');
        startwithdate(d);
    });

    $('#email').val(GetQueryString("email"));

    //bind click
    // $("#btn-time").bind("click",function(){
    //     alert('btn-time click');
    //     updateapp('time');
    // });
    //
    // $("#btn-have").bind("click",function(){
    //     alert('btn-have click');
    //     updateapp('have');
    // });
    //
    // $("#btn-no").bind("click",function(){
    //     alert('btn-no click');
    //     updateapp('no');
    // });
    //
    // $("#btn-partno").bind("click",function(){
    //     alert('btn-partno click ' );
    //     updateapp('partno');
    // });
    //
    // $("#btn-word-submit").bind("click",function(){
    //     var word_q = $("#word_q").val();
    //     var word_ay = $("#word_ay").val();
    //     var word_an = $("#word_an").val();
    //     var word_other = $("#word_other").val();
    //     updateword(word_q,word_ay,word_an,word_other);
    //     alert('保存成功 ' + word_q + ' ' + word_ay + ' ' + word_an + ' ' + word_other);
    // });


    // $("#messageform").on("submit", function() {
    //     newMessage($(this));
    //     return false;
    // });
    // $("#messageform").on("keypress", function(e) {
    //     if (e.keyCode == 13) {
    //         newMessage($(this));
    //         return false;
    //     }
    // });
    // $("#message").select();
    updater.start();
    console.log("do complete ready");
});

function toYYMMDDHH(t) {
    var d = new Date(t);
    console.log(d);
    var dd = d.getDate() < 10 ? "0" + d.getDate() : d.getDate().toString();
    var mmm = d.getMonth() +1 < 10 ? "0" + (d.getMonth()+1) : (d.getMonth()+1);
    var yyyy = d.getFullYear().toString();
    var hh = d.getHours() < 10 ? "0" + d.getHours() : d.getHours().toString();
    return yyyy + '-' + mmm + '-' + dd + ' ' + hh + ':00:00';
}


function ok(id){
    $('#' + id).hide();
    // toastr.success('都处理完成，请选择其他时间段');
    oknext();
}

function syncnew() {
    var url = '/predict?type=syncnew';
    $.get(url,function(data,status){
        data = JSON.parse(data);
        $('#total').html(data.total)
        $('#errors').html(data.error_total)
        $('#rate').html(data.rate)
        toastr.success('已同步最新数据');
    });
    $("#date").val('');
    var form = $('#messageform');
    newMessage(form);
}


function startwithdate(date) {
    $("#date").val(date);
    $("#errorids").val('');
    var form = $('#messageform');
    newMessage(form);
}

function GetQueryString(name)
{
    var reg = new RegExp("(^|&)"+ name +"=([^&]*)(&|$)");
    var r = window.location.search.substr(1).match(reg);
    if(r!=null)return  unescape(r[2]); return null;
}


function start() {
    $("#date").val('');
    $("#errorids").val('');
    var form = $('#messageform');
    newMessage(form);
}

function updateapp(type ,id) {
    //做更新操作
    var emailquery = GetQueryString("email") != '' ? '&email=' + GetQueryString("email") : '';
    var url = '/predict?type=' + type + '&id=' + id + emailquery;
    $.get(url,function(data,status){
        //toastr.success("Data: " + data + "\nStatus: " + status);
        data = JSON.parse(data);
        console.log(data);
        $('#total').html(data.total);
        $('#errors').html(data.error_total);
        $('#rate').html(data.rate);
        if(data.find != null) {
            toastr.success('恭喜，发现一个错误样本！准确率：' + data.rate + "%");
        } else {
            toastr.success('提交数据成功, 准确率：' + data.rate + "%");
        }
    });
    // var form = $('#messageform');
    // newMessage(form);
    // console.log(id);
    $('#' + id).hide();
    //添加到出错
    $('#errorlist').prepend('<li class="list-group-item" onclick="loaderror("' + id +'")">' + id + '</li>');
    oknext();
}

function oknext(){
    var form = $('#messageform');
    newMessage(form);
}

function updateword() {
    //做更新操作
    var word_q = $("#word_q").val();
    var word_ay = $("#word_ay").val();
    var word_an = $("#word_an").val();
    var word_other = $("#word_other").val();
    var url = '/predict?type=words&q=' + word_q + '&ay=' + word_ay + '&an=' + word_an + '&other=' + word_other;

    $.get(url,function(data,status){
        // toastr.success("Data: " + data + "\nStatus: " + status);
        //update
        data = JSON.parse(data);
        // console.log(data);

        $('#total').html(data.total)
        $('#errors').html(data.error_total)
        $('#rate').html(data.rate)

        toastr.success('提交数据成功, 准确率：' + data.rate + "%");

    });

}

// function setid(id) {
//     select_id = id;
//     $("#show"+id).html('<span style="color:red">选上</span>')
// }

function newMessage(form) {

    var totalcount = parseInt($('#total').html()) + 1;
    $("#total").html(totalcount);

    var no_total = parseInt($('#no_total').html());
    $("#no_total").html(no_total);

    var errorcount = parseInt($('#errors').html());
    if(no_total == 0){
        $("#rate").html('100');
    } else{
        $("#rate").html(( (1 - errorcount/no_total) * 100).toFixed(2));
    }

    $("#inbox").html('<div class="col-lg-12"></div>');
    var message = form.formToDict();
    // do socket send
    // console.log(updater.socket);
    updater.socket.send(JSON.stringify(message));
    form.find("input[type=text]").val("").select();
}

jQuery.fn.formToDict = function() {
    var fields = this.serializeArray();
    var json = {}
    for (var i = 0; i < fields.length; i++) {
        json[fields[i].name] = fields[i].value;
    }
    if (json.next) delete json.next;
    return json;
};

var updater = {
    socket: null,

    start: function() {
        var url = "ws://" + location.host + "/chatsocket";
        updater.socket = new WebSocket(url);
        updater.socket.onmessage = function(event) {
            var msg = JSON.parse(event.data)
            updater.showMessage(msg);
            // console.log(msg)

        }
        console.log("run start!");
    },

    showMessage: function(message) {
        if(message.id == null) {
            if(message.result != null) {
                toastr.info('这个时间段已经预测完成，请换一个时间段');
            } else {
                toastr.info('预测进行中,请稍后：' + message.process );
            }
        } else {
            var existing = $("#m" + message.id);
            if (existing.length > 0) return;
            var node = $(message.html);
            node.hide();
            $("#inbox").append(node);
            node.slideDown();
            var inboxDiv = document.getElementById("inbox");
            var div_s = inboxDiv.getElementsByTagName("section");
            toastr.success('加载数据中，已加载：' + div_s.length + '条');
        }
    }
};


function loaderror(ids)
{
    $("#date").val('');
    $("#errorids").val(ids);
    var form = $('#messageform');
    newMessage(form);
}

