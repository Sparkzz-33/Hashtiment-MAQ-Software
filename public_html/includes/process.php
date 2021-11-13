<?php


if (isset($_POST["hashtag"])) {
	$hashtag = $_POST["hashtag"];
	$limit = $_POST["limit"];
	$start = date('Y-m-d', strtotime($_POST["start_date"]));
	$end = date('Y-m-d', strtotime($_POST["end_date"]));
	$result = exec("py sentiment_file.py $hashtag $limit $start $end");
	$result = json_decode($result);
	echo "<img src = \"./includes/plot.png\" style = \"align-self:center;height:600px; width:900px;\">" ;
	exit();
}


?>