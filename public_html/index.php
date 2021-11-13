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
    <link href="https://fonts.googleapis.com/css2?family=Didact+Gothic&display=swap" rel="stylesheet">
    <style>
    	
    	body
    	{
    		background-image: url('./includes/bg3.jpg');
    		background-repeat: no-repeat;
		    background-attachment: fixed;
		    background-size: cover;
		    font-family: 'Didact Gothic', sans-serif;
    	}
    	.op_right
    	{
    		position: absolute;
    		right: 5%;
    		width: 33%;
    		border: 2px solid white;
    		top: 35%;
    		border-radius: 20px;
    		background-color: rgba(0,0,0,0.2)
    	}
    	.butt
    	{
    		height: 60px;
    		width: 80%;
    		margin-left: 32%
    		background-color: white;
    		display:flex;
    		border: 1px solid white;
    		margin: 2%;
    		border-radius: 10px;

    	}
    	.butt:hover
    	{
    		text-decoration-color: white; 
    		color: white;
    		background-color: #03a9fc;
    	}
    	.lin:hover
    	{
    		color: white;
    	}
    	.left_div
    	{
    		position: absolute;
    		left: 2%;
    		top: 30%;
    		color: white;
    	}
    	.nav_item:hover
		{
			color: #03a9fc;
		}
    </style>
</head>
<body>
	<?php  //Navigation Bar
		include_once("./templates/header.php");
	?>
	<div class="row">
	<div class="col-lg-7 left_div">
		<p style="font-size: 80px;">Welcome,</p>
		<p style="font-size: 60px;">This is an application to semantically analyze tweets</p>
	</div>
	<div class="op_right col-lg-5">
		<div class="butt mx-auto">
		<a href="hashtag.php" class="mx-auto lin"> <h2><i class="fa fa-hashtag"></i>&nbsp; Analyze Single Hashtag</h2> </a>
		</div>
		<br>
	</div>
	</div>
</body>
</html>