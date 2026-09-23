# biography-2418

function hideoverlay() {

  			document.getElementById("overlay").style.display ="none";
  		}
		function answer(){
			
			if (document.getElementById("provnum").innerHTML == '1') {
				
				if (document.getElementById('rad3').checked) {
				
					document.getElementById("overlay").innerHTML = "Yay, That Answer Was Correct! Try next the question. " ;
					
					document.getElementById("overlay").style.display = "initial" ;

					
					document.getElementById("provnum").innerHTML = '2';

					document.getElementById("provnum").style.color = "lime" ;
					
					document.getElementById('rad3').checked = false;
					return;
				}
				
				else {
				
					document.getElementById("overlay").innerHTML = "Sorry, your answer seemed to be incorrect, please try again.  ";
					
					document.getElementById("overlay").style.display = "initial" ;
					
					var ele = document.getElementsByName("prov");
   						
   						for(var i=0; i
