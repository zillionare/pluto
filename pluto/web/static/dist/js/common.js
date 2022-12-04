function today(){
    return new Date().toISOString().slice(0, 10)
}

function bind_enter_key(sel, f){
    $(sel).keypress(function(e){
        if (e.keyCode == 13) {
            f($(this).val())
        }
        return false
    })
}
