<%inherit file="lkc_bisnp.web:templates/base.mako" />

<p>This is an online tool to predict country of origin of a sample based on likelihood classifier.
    Countries used to train the classifier are: Afghanistan, Bangladesh, Bhutan, Brazil, Cambodia, China,
    Colombia, Ethiopia, India, Indonesia, Iran, Madagascar, Malaysia, Mexico, Myanmar, Papua New Guinea, Peru, Sudan,
    Thailand, Vietnam.

<form action='/run' method='POST'>
    <div class="form-group">
        <label for="BarcodeData">Please enter the data with the following format: SNP_Barcode&lt;tab&gt;Sample_ID</label>
        <textarea class="form-control rounded-0" id="BarcodeData" name="BarcodeData" rows="10" style="font-family: 'Cousine', monospace;">
CATAGTGCAGAATCTGCTTTCCATCCGA	cambodia1
TCTGGTGCAGAATCTGTTTTCCATGGGA	thai1
CCTAGTGCAGCAGCTTCTTTCCATCGGC	thai2
TCTGTTGCAAAATCTGTCCTTCTTCCGA	PNG1
TCGAGTGCAGCATCTGCTTCCATTCCGA	india1
CATAGXGCAGAAXCTGCTXTCCAXCCGX	cambodia1_miss5
CCTAGTGCGCAGCTTCTTTCCATCGGC	thai3_wrongcode
        </textarea>
    </div>

    <div class="form-group">
        <label for="SNPSet">SNP Set</label>
        <select class="form-control" id="SNPSet" name="SNPSet">
            <option value="SNP-33">SNP-33</option>
            <option value="Broad-37">Broad 37 SNPs</option>
            <option value="SNP-28+Broad">SNP-28+Broad (65 SNPs)</option>
            <option value="SNP-51">SNP-51 (HFST)</option>
            <option value="SNP-50">SNP-50 (DecisionTree)</option>
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
