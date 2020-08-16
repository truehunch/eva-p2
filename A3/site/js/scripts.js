/*!
    * Start Bootstrap - Freelancer v6.0.4 (https://startbootstrap.com/themes/freelancer)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-freelancer/blob/master/LICENSE)
    */
    (function($) {
    "use strict"; // Start of use strict
  
    // Smooth scrolling using jQuery easing
    $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
      if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
        var target = $(this.hash);
        target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
        if (target.length) {
          $('html, body').animate({
            scrollTop: (target.offset().top - 71)
          }, 1000, "easeInOutExpo");
          return false;
        }
      }
    });
  
    // Scroll to top button appear
    $(document).scroll(function() {
      var scrollDistance = $(this).scrollTop();
      if (scrollDistance > 100) {
        $('.scroll-to-top').fadeIn();
      } else {
        $('.scroll-to-top').fadeOut();
      }
    });
  
    // Closes responsive menu when a scroll trigger link is clicked
    $('.js-scroll-trigger').click(function() {
      $('.navbar-collapse').collapse('hide');
    });
  
    // Activate scrollspy to add active class to navbar items on scroll
    $('body').scrollspy({
      target: '#mainNav',
      offset: 80
    });
  
    // Collapse Navbar
    var navbarCollapse = function() {
      if ($("#mainNav").offset().top > 100) {
        $("#mainNav").addClass("navbar-shrink");
      } else {
        $("#mainNav").removeClass("navbar-shrink");
      }
    };
    // Collapse now if page is not at top
    navbarCollapse();
    // Collapse the navbar when page is scrolled
    $(window).scroll(navbarCollapse);
  
    // Floating label headings for the contact form
    $(function() {
      $("body").on("input propertychange", ".floating-label-form-group", function(e) {
        $(this).toggleClass("floating-label-form-group-with-value", !!$(e.target).val());
      }).on("focus", ".floating-label-form-group", function() {
        $(this).addClass("floating-label-form-group-with-focus");
      }).on("blur", ".floating-label-form-group", function() {
        $(this).removeClass("floating-label-form-group-with-focus");
      });
    });
  
  })(jQuery); // End of use strict

// Register the plugin
FilePond.registerPlugin(FilePondPluginImagePreview);

const inputElement1 = document.getElementById('filepond-a1');
const pond_a1 = FilePond.create( inputElement1 );

function upload_to_lambda_a1(fieldName, file, metadata, load, error, progress, abort) {
  $('#result-a1').css('visibility', 'hidden');
  var formData = new FormData();
  formData.append(file.name, file)

  progress(true, 0.25, 1);
  // Custom ajax file upload or local storing here
  $.ajax({
    async: true,
    crossDomain: true,
    tryCount: 0,
    retryLimit : 3,
    method: 'POST',
    url: 'https://c1ad8k6is0.execute-api.ap-south-1.amazonaws.com/dev/assignment2',
    data: formData,
    processData: false,
    contentType: false,
    mimeType: "multipart/form-data",
    timeout: 5000,
    fail: (xhr, textStatus, errorThrown) => error(errorThrown)
  })
  .done((response) => {
    console.log(response)
    $('#result-a1').css('visibility', 'visible');
    $('#result-a1').attr('class', 'alert alert-success');
    $('#result-a1-prediction').empty()
    response = JSON.parse(response)
    $('#result-a1-prediction').append('<div class="text-left">')
      Object.keys(response).forEach((key) => {
        $('#result-a1-prediction').append(
          '<p>' 
            + '<h5 class="text-uppercase">' + key + '</h5>' + response[key] 
        + '</p>');
    });
    $('#result-a1-prediction').append('</div>')
  })
  .fail(() => {
    $('#result-a2').css('visibility', 'visible')
    $('#result-a2').attr('class', 'alert alert-danger');
    $('#result-a2-prediction').text('Operation failed, please retry')
  })
  progress(true, 1, 1);
  load('unique-file-id');
  return {
    abort: function() {
      abort();
    }
  };
}

pond_a1.setOptions({
  allowReplace: false,
  instantUpload: false,
  maxFiles: 1,
  server: {
    process: upload_to_lambda_a1,
    revert: null
  }});

$('#result-a1').css('visibility', 'hidden');

const inputElement2 = document.getElementById('filepond-a2');
const pond_a2 = FilePond.create( inputElement2 );

function upload_to_lambda_a2(fieldName, file, metadata, load, error, progress, abort) {
  $('#result-a1').css('visibility', 'hidden');
  var formData = new FormData();
  formData.append(file.name, file)

  progress(true, 0.25, 1);
  // Custom ajax file upload or local storing here
  $.ajax({
    async: true,
    crossDomain: true,
    tryCount: 0,
    retryLimit : 3,
    method: 'POST',
    url: 'https://ob38c41f22.execute-api.ap-south-1.amazonaws.com/dev/assignment2',
    data: formData,
    processData: false,
    contentType: false,
    mimeType: "multipart/form-data",
    timeout: 5000,
    fail: (xhr, textStatus, errorThrown) => error(errorThrown)
  })
  .done((response) => {
    $('#result-a2').css('visibility', 'visible');
    $('#result-a2').attr('class', 'alert alert-success');
    $('#result-a2-prediction').empty()
    response = JSON.parse(response)
    console.log(response)

    $('#result-a2-prediction').append(
      '<p>' 
        + '<h5 class="text-uppercase">' + 'file' + '</h5>' + response['file'] 
    + '</p>');

    $('#result-a2-prediction').append('<div class="text-left">')
      Object.keys(response['predicted']).forEach((key) => {
        $('#result-a2-prediction').append(
          '<p>' 
            + '<h5 class="text-uppercase">' + key + '</h5>' + response['predicted'][key].toString()
        + '</p>');
    });
    $('#result-a2-prediction').append('</div>')
  })
  .fail(() => {
    $('#result-a2').css('visibility', 'visible')
    $('#result-a2').attr('class', 'alert alert-danger');
    $('#result-a2-prediction').text('Operation failed, please retry')
  })
  progress(true, 1, 1);
  load('unique-file-id');
  return {
    abort: function() {
      abort();
    }
  };
}

pond_a2.setOptions({
  allowReplace: false,
  instantUpload: false,
  maxFiles: 1,
  server: {
    process: upload_to_lambda_a2,
    revert: null
  }});

$('#result-a2').css('visibility', 'hidden');

const inputElement = document.getElementById('filepond-a3');
const pond_a3 = FilePond.create( inputElement );


function upload_to_lambda_a3(fieldName, file, metadata, load, error, progress, abort) {
  $('#result-a3').css('visibility', 'hidden');
  var formData = new FormData();
  formData.append(file.name, file)

  progress(true, 0.25, 1);

  axios.post('https://2u5go6le4m.execute-api.ap-south-1.amazonaws.com/dev/assignment3-face-align', 
             formData,
             {responseType: 'blob'})
  .then(function (response) {
    console.log(response)
    //var arrayBufferView = new Uint8Array( response['data'] );
    //var blob = new Blob( [ arrayBufferView ], { type: "image/png" } );
    var URL = window.URL || window.webkitURL;
    blob = response['data']
    var imageUrl = URL.createObjectURL( blob );
    console.log(imageUrl)
    
    var img = new Image(100, 100);
    // So the Blob can be Garbage Collected
    img.onload = e => URL.revokeObjectURL( imageUrl );
    img.src = imageUrl;

    $('#result-a3').css('visibility', 'visible')
    document.querySelector("#result-a3-prediction-img").src = imageUrl;
  })
  .catch(function (error) {
    console.log(error);
  });

  progress(true, 1, 1);
  load('unique-file-id');
  return {
    abort: function() {
      abort();
    }
  };
}

pond_a3.setOptions({
  allowReplace: false,
  instantUpload: false,
  maxFiles: 1,
  server: {
    process: upload_to_lambda_a3,
    revert: null
  }});

$('#result-a3').css('visibility', 'hidden');