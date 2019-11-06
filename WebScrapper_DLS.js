var obj = [[200, 15, 2, 20, 100]];
var x = JSON.parse(obj);
var y = []
var dl_score;
function prep3() {
	$.each(x, function( index, value ) {
		$("#myn1").val(value[0]);
		$("#myn2").val(value[1]);
		$("#myn3").val(value[2]);
		$("#myn4").val(value[3]);
		$("#myn5").val(value[4]);
		var team_score = parseInt(value[5]);
		$("input[value='Calculate']").trigger("click");
		var i = $( "td:contains('The Target')" ).text().split(" ");
		dl_score = parseInt(i[i.length-1]);
		if (team_score >= dl_score) {
			y.push(1);
		} else {
			y.push(0);
		}
	});
}