<%inherit file="lkc_bisnp.web:templates/base.mako" />

<p>Results in the format of [probability] country.<br>Classifier: ${code}</p>

<table class='table table-striped table-sm'>

	<thead class='thead-dark'>
		<tr><th>Sample</th><th>1st Prediction</th><th>2nd Prediction</th><th>3rd Prediction</th>
		<th>Hets</th><th>Miss</th><th>Masked</th><th>Note</th></tr>
	</thead>

	<tbody>
		% for line in table:
		<tr>
			${ ''.join( ['<td>{}</td>'.format(x) for x in line] ) | n}
		</tr>
		% endfor
	</tbody>

</table>
