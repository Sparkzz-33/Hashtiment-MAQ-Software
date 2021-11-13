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
    <style>
    	.nav_item:hover
	{
		color: #03a9fc;
	}
    </style>
</head>
<body>
	<?php
		include_once("./templates/header.php");
	?>
		<div class="row">
			<div class="container col-md-8" id="out_plot1" name="out_plot1" style="align-self: center; align-content: center" class="mx-auto">
				<img src="./includes/plot.png">
			</div>
			
			<div class="col-md-4">
				<br><br><br><br>
				<h3>Parameters</h3>
				<h4>Hashtag : <?php   
					$f = fopen("./includes/demo.txt", "r") or die("Unable to open");
					echo fgets($f);?>
				</h4>
				<h4>Limit : 
					<?php
						echo fgets($f);			
					?>
				</h4>
				<h4>Start-date:
					<?php
						echo fgets($f);			
					?>
				</h4>
				<h4>End-date:
					<?php
						echo fgets($f);			
					?>
				</h4>
			</div>
		</div>
		<br>
		<div class="container mx-auto">
			<button class="btn btn-primary go_back"> Search Other Query </button>
		</div>
</body>
</html>