<html>
    <body>
    <header>
            <a href="index.php" class="logo">MAIA</a>

            <ul>
                <li>
                    <a href="#">Products</a>
                    <ul class="dropdown">
                        <li><a href="newproducts.php" class="ddp">New Products</a><hr></li>
                        <li><a href="lipbalms.php" class="ddp">Lip Balms</a><hr></li>
                        <li><a href="hairessentials.php" class="ddp">Hair Essentials</a><hr></li>
                        <li><a href="facecare.php" class="ddp">Face Care</a><hr></li>
                        <li><a href="bodycare.php" class="ddp">Body Care</a><hr></li>
                    </ul>
                
                </li>


                <li>
                    <a href="Sale.php">Sale</a>
                </li>
                <li><a href="booking.php">Booking</a></li>
                <li><a href="#"><img src="../images/line.png" width="30px"></a></li>
                <li><a href="#"><img src="../images/fb.png" width="35px" ></a></li>
                <li><a href="#"><img src="../images/insta.png" width="30px"></a></li>
                <li><a href="#"><img src="../images/twitter.png" width="30px"></a></li>
                <li><a href="#"><img src="../images/line.png" width="30px"></a></li>
                <li><a href="cart.php"><img src="../images/cart.png" width="35px"></a></li>
                <li><a href="#"><img src="../images/search.png" width="30px"></a></li>
                <li><a href="#"><img src="../images/img3.png" width="30px"></a>
               
                <ul class="dropdown">
                    <li><a href="profile.php" class="ddp">Profile</a><hr></li>
                    <li><a href="#" class="ddp">LogOut</a><hr></li>
                    
                </ul>
            
                </li>
            </ul>
            </header>

            <section class="banner"></section>
            <script type="text/javascript">
                window.addEventListener("scroll", function(){
                var header = document.querySelector("header");
                header.classList.toggle("sticky", window.scrollY > 0);
                })
            </script>
            
    </body>

</html>
