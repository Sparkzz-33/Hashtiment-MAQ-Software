$(document).ready(function(){
	var DOMAIN = "http://localhost/hashtiment-master/public_html";


	$("#form_hashtag").on("submit", function(){
		$("#btn_submit").attr("disabled", true);
		var hashtag = $("#hashtag");
		//var start_date, end_date;
		//console.log(start);
		//document.getElementById('loader').style.display='block';
		$.ajax({
			url : DOMAIN + "/includes/process.php",
			method : "POST",
			data : $('#form_hashtag').serialize(),
			success : function(data) {
				console.log(data)
				window.location.href = "output.php";
			}
		})
	})

	$("body").delegate(".go_back", "click", function(){
		window.location.href = "index.php?";
	})
})