<%inherit file="lkc_bisnp.web:templates/base.mako" />

<p>This is an online tool to predict country of origin of a sample based on likelihood classifier.
    Countries used to train the classifier are: Afghanistan, Bangladesh, Bhutan, Brazil, Cambodia, China,
    Colombia, Ethiopia, India, Indonesia, Iran, Madagascar, Malaysia, Mexico, Myanmar, Papua New Guinea, Peru, Sudan,
    Thailand, Vietnam.

<form action='/run' method='POST' enctype="multipart/form-data">
    <div class="mb-3">
        <label for="BarcodeData">Please enter the data according to Data Format:</label>
        <textarea class="form-control rounded-0" id="BarcodeData" name="BarcodeData" rows="10" style="font-family: 'Cousine', monospace;">
CGGTTAATTACCGGGGGCACGGCCTGCTGCGCA	Colombia-1
CGTTTAGGTTTCGTGGGCACGGCCAGCAGCGTT	Myanmar-1
TAGCTGGATCGTTGAGGTCGAGTCAGCAGCGCT	Mismatch_Barcode
CGGCTAATTACCGTGGCACGGCCTGCATCGCA	Wrong_BarcodeLength
        </textarea>
    </div>

    <div class="mb-3">
        <b>- or -</b>
    </div>

    <div class="mb-3">
        <label for="InFile" class="form-label">Input file</label>
        <input class="form-control" type="file" id="InFile" name="InFile">
    </div>

    <div class="mb-3">
        <label for="DataFormat">Data Format</label>
        <select class="form-select" id="DataFormat" name="DataFormat">
            <%! from lkc_bisnp.web.lib.utils import data_formats %>
            % for key, text in data_formats:
                <option value="${key}">${text}</option>
            % endfor
        </select>
    </div>

    <div class="mb-3">
        <label for="Classifier">Classifier</label>
        <select class="form-select" id="Classifier" name="Classifier">
            % for key, clf in sorted(classifiers.items()):
                <option value="${key}">${clf[0]}</option>
            % endfor
        </select>
    </div>

    <div class="mb-3">
        <button type="submit" class="btn btn-primary">Run classifier</button>
    </div>

</form>

<script src="/static/assets/js/behave.js"></script>
<script type="text/javascript">
//<![CDATA[

var editor = new Behave({
    textarea: document.getElementById('BarcodeData'),
    replaceTab: true,
    softTabs: false,
    tabSize: 8
});

//]]>
</script>
