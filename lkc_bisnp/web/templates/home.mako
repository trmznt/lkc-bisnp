<%inherit file="lkc_bisnp.web:templates/base.mako" />

<p>This is an online tool to predict country of origin of a sample based on likelihood classifier.
    Countries used to train the classifier are: Afghanistan, Bangladesh, Bhutan, Brazil, Cambodia, China,
    Colombia, Ethiopia, India, Indonesia, Iran, Madagascar, Malaysia, Mexico, Myanmar, Papua New Guinea, Peru, Sudan,
    Thailand, Vietnam.

<form action='/run' method='POST'>
    <div class="form-group">
        <label for="BarcodeData">Please enter the data with the following format: SNP_Barcode&lt;tab&gt;Sample_ID</label>
        <textarea class="form-control rounded-0" id="BarcodeData" name="BarcodeData" rows="10" style="font-family: 'Cousine', monospace;">
CGGTTAATTACCGGGGGCACGGCCTGCTGCGCA	Colombia-1
CGTTTAGGTTTCGTGGGCACGGCCAGCAGCGTT	Myanmar-1
TAGCTGGATCGTTGAGGTCGAGTCAGCAGCGCT	Mismatch_Barcode
CGGCTAATTACCGTGGCACGGCCTGCATCGCA	Wrong_BarcodeLength
        </textarea>
    </div>

    <div class="form-group">
        <label for="Classifier">Classifier</label>
        <select class="form-control" id="Classifier" name="Classifier">
            <option value="0">Biallele Likelihood with SNP-33</option>
            <option value="1">Biallele Likelihood with 72 SNPs (31 GEO + 41 VN Antwerp)</option>
            <option value="2">BernoulliNB with Broad-37 SNPs (hets treated as missing SNPs)</option>
        </select>
    </div>

    <button type="submit" class="btn btn-primary">Run classifier</button>

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
