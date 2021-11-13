<!DOCTYPE html>
<html>
<head>
	<title>Home</title>
	<meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">
    <script type="text/javascript" src="./js/main.js"></script>

    <script>
    function loadingCircle() {
        $("#loader").show();
        }

    </script>

    <style>
    	body
    	{
    		background-image: url('./includes/bg3.jpg');
    		background-repeat: no-repeat;
		    background-attachment: fixed;
		    background-size: cover;
		    font-family: 'Didact Gothic', sans-serif;
    	}
/* Center the loader */
	#loader {
	  position: absolute;
	  left: 50%;
	  top: 50%;
	  z-index: 1;
	  width: 90px;
	  height: 90px;
	  margin: -75px 0 0 -75px;
	  border: 10px solid #f3f3f3;
	  border-radius: 50%;
	  border-top: 10px solid #03a9fc;
	  -webkit-animation: spin 2s linear infinite;
	  animation: spin 2s linear infinite;
	}

	@-webkit-keyframes spin {
	  0% { -webkit-transform: rotate(0deg); }
	  100% { -webkit-transform: rotate(360deg); }
	}

	@keyframes spin {
	  0% { transform: rotate(0deg); }
	  100% { transform: rotate(360deg); }
	}
	.nav_item:hover
	{
		color: #03a9fc;
	}
	.cb
	{
		background-color: rgba(0,0,0,0.7);
		color: #03a9fc;
		font-weight: 600;
	}
	#h_error
	{
		color: white;
	}

	</style>
</head>
<body>
	<?php  //Navigation Bar
		include_once("./templates/header.php");
	?>
	<br><br><br><br>
	<div class="card mx-auto cb" style="width: 25rem; height: 30rem">
	  	<!-- <img src="./images/login.jpg" class="card-img-top mx-auto"> -->
		 	 <div class="card-body">
		 	 	<form id="form_hashtag" onsubmit="return false" autocomplete="off">
				  <div class="form-group">
				    <label for="hashtag">Enter Hashtag</label>
				    <input type="text" class="form-control" id="hashtag" name="hashtag" style="background-color: rgba(0,0,0,0.5); color: white">
				    <small id="h_error" class="form-text" style="text-decoration-color: white;">Enter your hashtag here</small>
				  </div>
				  <br>
				  <div class="form-group">
				  	<label for="limit">Max tweets to be analyzed</label>
				  	<input type="text" name="limit" id="limit" class="form-control" style="background-color: rgba(0,0,0,0.5); color: white" placeholder="Maximum 1000 tweets can be analyzed">
				  </div>
				  <br>
				  <div class="form-row">
				  	<div class="form-group col-md-6">
				  		<label>Start Date</label>
				  		<input type="date" class="form-control" name="start_date">
				  	</div>
				  	<div class="form-group col-md-6">
				  		<label>End Date</label>
				  		<input type="date" class="form-control" name="end_date">
				  	</div>
				  </div>
				  <br>
				  <button type="submit" onclick="loadingCircle()" class="btn btn-primary" id="btn_submit">  Submit</button>
				</form>
				<div id="loader" style="display: none;"></div>
		 	</div>	
		</div>
		<br><br>
</body>
</html>