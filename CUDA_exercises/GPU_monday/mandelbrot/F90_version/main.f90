program main

!*****************************************************************************80
!
!  MAIN is the main program for MANDELBROT.
!
!  Discussion:
!
!    MANDELBROT computes an image of the Mandelbrot set.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    08 August 2009
!
!  Author:
!
!    John Burkardt
!
!  Modified by:
!    Bernd Dammann
!    Boyan Lazarov
!
!  Local Parameters:
!
!    Local, integer COUNT_MAX, the maximum number of iterations taken
!    for a particular pixel.
!
  implicit none

  integer   ( kind = 4 ) :: n = 2501
  integer   ( kind = 4 ) :: count_max = 800

  integer   ( kind = 4 ) :: c
  real      ( kind = 8 ) :: c_max, c_max_inv
  integer   ( kind = 4 ), dimension(:,:), allocatable :: image
  character ( len = 255 ) :: filename 

  real      ( kind = 8 ) :: x_max =   1.25D+00
  real      ( kind = 8 ) :: x_min = - 2.25D+00
  real      ( kind = 8 ) :: y_max =   1.75D+00
  real      ( kind = 8 ) :: y_min = - 1.75D+00

  allocate(image(n,n))

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'MANDELBROT'
  write ( *, '(a)' ) '  FORTRAN90 version'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  Create an PNG image of the Mandelbrot set.'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  For each point C = X + i*Y'
  write ( *, '(a,g14.6,a,g14.6,a)' ) '  with X range [', x_min, ',', x_max, ']'
  write ( *, '(a,g14.6,a,g14.6,a)' ) '  and  Y range [', y_min, ',', y_max, ']'
  write ( *, '(a,i8,a)' ) '  carry out ', count_max, ' iterations of the map'
  write ( *, '(a)' ) '  Z(n+1) = Z(n)^2 + C.'
  write ( *, '(a)' ) '  If the iterates stay bounded (norm less than 2)'
  write ( *, '(a)' ) '  then C is taken to be a member of the set.'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  A PNG image of the set is created using'
  write ( *, '(a,i8,a)' ) '    N = ', n, ' pixels in the X direction and'
  write ( *, '(a,i8,a)' ) '    N = ', n, ' pixels in the Y direction.'
  write ( *, '(a)' ) ' '
!
!  Carry out the iteration for each pixel, determining COUNT.

 call timestamp ( )

 call mandel(n,image,count_max)

 write ( *, '(a)' ) ' '
 write ( *, '(a)' ) ' Calculation of the image finished. '
 call timestamp ( )

! uncomment the following line, if you don't need the PNG output
!  stop
!
! call writepng to save a PNG image in filename

  filename = "mandelbrot.png"//CHAR(0)
  call writepng(filename, image, n, n)
  deallocate(image)

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) &
    '  PNG image data stored in "' // trim ( filename ) // '".'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'MANDELBROT'
  write ( *, '(a)' ) '  Normal end of execution.'
  write ( *, '(a)' ) ' '
  call timestamp ( )

  stop
end
