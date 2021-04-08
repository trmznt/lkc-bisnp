<%inherit file="lkc_bisnp.web:templates/base.mako" />

<p>Results in the format of [probability] country.</p>

<table class='table table-striped table-sm'>

	<thead class='thead-dark'>
		<tr><th>Sample</th><th>1st Prediction</th><th>2nd Prediction</th><th>3rd Prediction</th></tr>
	</thead>

	<tbody>
		% for line in table:
		<tr>
			${ ''.join( ['<td>{}</td>'.format(x) for x in line] ) | n}
		</tr>
		% endfor
	</tbody>

</table>

% if len(logs) > 0:

<p><b>Log Report</b></p>
<ul>
	% for log in logs:
	<li>${log}</li>
	% endfor
</ul>

% endif
