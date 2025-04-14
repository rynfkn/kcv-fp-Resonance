$(document).ready(function() {
    // Image preview
    $('#image').change(function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                $('#imagePreview').attr('src', e.target.result).show();
            }
            reader.readAsDataURL(file);
        }
    });
    
    // City change handler - can be used to filter housing options
    $('#City').change(function() {
        const cityId = $(this).val();
        const housingSelect = $('#Nama_Perumahan');
        
        // Disable select while filtering (can be expanded with actual filtering logic)
        housingSelect.prop('disabled', true);
        setTimeout(() => {
            housingSelect.prop('disabled', false);
        }, 300);
    });
    
    // Form submission with your provided API call
    $('#predictionForm').submit(function(e) {
        e.preventDefault();
        
        // Hide previous results and show progress
        $('#resultBox').hide();
        $('#errorBox').hide();
        $('.progress').show();
        
        // Create form data
        const formData = new FormData(this);
        
        // Make AJAX request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                console.log("Success response:", response);
                $('.progress').hide();
                
                let price;
                if (Array.isArray(response.predicted_price_millions)) {
                    price = response.predicted_price_millions[0];
                } else {
                    price = response.predicted_price_millions;
                }
                
                $('#prediction').text('IDR ' + response.predicted_price_millions[0]);
                
                $('#resultBox').show();
            },
            error: function(xhr) {
                $('.progress').hide();
                const errorData = JSON.parse(xhr.responseText);
                $('#errorMessage').text(errorData.error);
                $('#errorBox').show();
            }
        });
    });
});